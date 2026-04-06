"""
For the 4 LLM pairs (we dropped 2 where deepseek was verifier):
                                                a. DeepSeek - GPT
                                                b. GPT - Qwen
                                                c. Qwen - GPT
                                                d. DeepSeek - Qwen  (justifications to be taken from .csv)

1. Sample randomly 25 TN explanations from natural space & 25 TN from mutated space
    a. listdir in TN/ folder to get the len of .txts
    b. run "get_numpy_rng" to get rng
    c. sample 25 random indices from each space & load them
    d. verifier_metadata.txt as a list of dicts to get verifier settings (or something custom)
    e. based on the file names, get the corresponding verifier justification, epr expl & epr ver, conf
    f. map the sample to its original xai-method-output so we can grab it (something like "detect_xai_method_from_explanation" function from run_new_verifier.py)
    g. so now we have xai-method-output, problematic explanation, verifier-justification -> place them in template
    
2. Run our architecture until explanation gets verified
    a. prompt the explainer with the newly formed template to get new explanation and feed it to verifier passing a parameter if it's first round or not
    b. if verified, stop, else, ++ rounds of reffeed, refeed the new explanation to explainer with the same template but updated explanation
    
3. Measure rounds of refeeding needed to finally verify the explanation
    a. keep a counter of rounds until verified

4. Save EPR history for both explainer & verifier
    a. for each sample, save the EPR history of both explainer & verifier

5. ...Plot
    a. present samples as violins
    b. lines plot for EPR history
"""
import json
import yaml
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from io import StringIO
from time import perf_counter as pcnt
import re
import datetime

# ============================================================================
# Configuration
# ============================================================================

SEED = 55
NATURAL_TN_SAMPLES = 35
MAX_REFEED_ITERATIONS = 10  # Stop after this many attempts

@dataclass
class ExperimentConfig:
    
    # LLM pair paths (explainer_verifier format)
    llm_pairs: Dict[str, Dict[str, Path]] = field(default_factory=lambda: {
        "deepseek_gpt": {
            "tn_path": Path("..") / "explainer" / "results" / "explanations_space_deep_gpt" / "processed_results" / "TN",
            "metadata_path": Path("..") / "explainer" / "results" / "explanations_space_deep_gpt" / "verifier_metadata.txt",
            # "results_folder": Path("..") / "explainer" / "results" / "explanations_space_deep_gpt"
        },
        "gpt_qwen": {
            "tn_path": Path("..") / "explainer" / "results" / "explanations_space_gpt_qwen" / "processed_results" / "TN",
            "metadata_path": Path("..") / "explainer" / "results" / "explanations_space_gpt_qwen" / "verifier_metadata.txt",
            # "results_folder": Path("..") / "explainer" / "results" / "explanations_space_gpt_qwen"
        },
        "qwen_gpt": {
            "tn_path": Path("..") / "explainer" / "results" / "explanations_space_qwen_gpt" / "processed_results" / "TN",
            "metadata_path": Path("..") / "explainer" / "results" / "explanations_space_qwen_gpt" / "verifier_metadata.txt",
            # "results_folder": Path("..") / "explainer" / "results" / "explanations_space_qwen_gpt"
        },
        "deep_qwen": {
            "tn_path": Path("..") / "explainer" / "results" / "explanations_space_deep_gpt" / "processed_results" / "TN",
            "metadata_path": Path("..") / "explainer" / "results" / "verifier_test_results_deep_qwen" / "verifier_test_results.csv",        # special case for deepseek-qwen
        }
    })
    
    # Mutation space base path
    mutated_space_path_base : Path = Path(".." ) / "explainer" / "results" / "mutated_space"
    mutation_space: List[Path] = field(default_factory=list)
    
    
    
    # XAI method outputs base path
    xai_outputs_base: Path = Path("..") / ".." / "XAI-Methods-outputs"
    
    # Architecture config
    arch_config_path: Path = Path("..") / "architecture.yaml"
    
    # Output path for experiment results
    output_path: Path = Path("feedback_experiment_results")
    
    def __post_init__(self):
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.mutation_space = {
            ds.name: {
                "explanations": list(ds.rglob("*_mut*.txt")),
                "metadata": list(ds.rglob("metadata.txt"))
            }
            for ds in self.mutated_space_path_base.iterdir()
            if ds.is_dir()
        }
        


# ============================================================================
# Helper Functions
# ============================================================================

def load_architecture_config(config_path: Path) -> dict:
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_numpy_rng(seed: int) -> np.random.Generator:
    
    return np.random.default_rng(seed)


def calculate_EPR(logprobs_info) -> float:
    
    if not logprobs_info or not isinstance(logprobs_info, list):
        return 0.0
    
    Lq = len(logprobs_info)
    log_probs_matrix = np.full((Lq, 10), np.nan, dtype=np.float32)

    for i, logprob_obj in enumerate(logprobs_info):
        if logprob_obj and "top_logprobs" in logprob_obj and logprob_obj["top_logprobs"]:
            vals = [item["logprob"] for item in logprob_obj["top_logprobs"]]
            n = min(len(vals), 10)
            log_probs_matrix[i, :n] = vals[:n]
    
    if np.all(np.isnan(log_probs_matrix)):
        return 0.0
        
    log_probs = np.atleast_2d(log_probs_matrix)
    lp_max = np.nanmax(log_probs, axis=1, keepdims=True)
    probs = np.exp(log_probs - lp_max)
    probs = np.nan_to_num(probs, nan=0.0)
    prob_sums = np.sum(probs, axis=1, keepdims=True)
    prob_sums[prob_sums == 0] = 1.0
    probs /= prob_sums
    
    entropies = -np.sum(probs * np.log2(probs + 1e-10), axis=1)
    epr = float(np.mean(entropies))
    
    return 0.0 if np.isnan(epr) else epr


def detect_xai_method_from_explanation(explanation_text: str) -> Tuple[str, Path]:
    
    xai_base = Path("..") / ".." / "XAI-Methods-outputs"
    
    # Keyword-based detection
    keywords_map = {
        "EBM": (["Sulphates", "sulphates", "alcohol", "volatile acidity"], 
                xai_base / "Wine" / "wine.EBM.out.json"),
        "Grad-Cam++": (["Middle-Center", "middle-center", "heatmap"], 
                       xai_base / "CIFAR10" / "gradcampp_sample2_output.json"),
        "SHAP": (["WKHP", "AGEP", "SCHL"], 
                 xai_base / "ACS_Income" / "shap_acs_output.json"),
        "Integrated Gradients": (["films", "film", "movie", "sentiment"], 
                                  xai_base / "IMDB" / "int_gradients_sample198_output.json"),
        "LIME": (["Carat", "carat", "diamond", "price"], 
                 xai_base / "Diamonds" / "LIME_diamonds.out.json")
    }
    
    explanation_lower = explanation_text.lower()
    
    for method, (keywords, path) in keywords_map.items():
        if any(keyword.lower() in explanation_lower for keyword in keywords):
            return method, path
    
    assert False, "Could not detect XAI method from explanation. Should not happen."


def extract_CoT_answer(full_output: str) -> Dict[str, str]:
    
    start_tag = "<think>"
    end_tag = "</think>"
    
    if start_tag in full_output and end_tag in full_output:
        start_pos = full_output.find(start_tag) + len(start_tag)
        end_pos = full_output.find(end_tag)
        cot = full_output[start_pos:end_pos].strip()
        answer = full_output[end_pos + len(end_tag):].strip()
    else:
        cot = ""
        answer = full_output
    
    return {
        "CoT": cot,
        "answer": answer,
        "raw_output": full_output
    }


# ============================================================================
# LLM Query Functions
# ============================================================================

def query_explainer_stream(arch_config: dict, prompt: str, model_override: str = None) -> Tuple[str, float]:

    model = model_override or arch_config["Explainer"]["model"]
    url = f"{arch_config['host']}:{arch_config['port']}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": arch_config["Explainer"].get("temperature", 2.5),
        "top_p": arch_config["Explainer"].get("top_p", 1.0),
        "top_k": arch_config["Explainer"].get("top_k", 0),
        "stream": True,
        "think": model not in ["phi4-reasoning:latest"],
        "logprobs": True,
        "top_logprobs": 10,
        "options": {
            "num_ctx": arch_config["Explainer"].get("num_ctx", 131072)  # 128k context
        }
    }
    
    response = requests.post(url, json=payload, stream=True)
    response.raise_for_status()
    
    thinking_buffer = StringIO()
    answer_buffer = StringIO()
    all_logprobs = []
    
    for line in response.iter_lines():
        if not line:
            continue
        try:
            data = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            continue
        
        if thinking_text := data.get("thinking", ""):
            thinking_buffer.write(thinking_text)
        if answer_text := data.get("response", ""):
            answer_buffer.write(answer_text)
        if "logprobs" in data and data["logprobs"]:
            all_logprobs.extend(data["logprobs"])
    
    full_output = f"<think>{thinking_buffer.getvalue()}</think>\n{answer_buffer.getvalue()}"
    epr = calculate_EPR(all_logprobs)
    
    return full_output, epr


def query_verifier(arch_config: dict, prompt: str, model_override: str = None) -> Tuple[str, float]:
    
    model = model_override or arch_config["Verifier"]["model"]
    url = f"{arch_config['host']}:{arch_config['port']}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": arch_config["Verifier"].get("temperature", 0.6),
        "stream": False,
        "think": model not in ["phi4-reasoning:latest"],
        "logprobs": True,
        "top_logprobs": 10,
        "options": {
            "num_ctx": arch_config["Verifier"].get("num_ctx", 131072)
        }
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    result = response.json()
    
    logprobs = result.get("logprobs", [])
    epr = calculate_EPR(logprobs) if logprobs else 0.0
    
    return result["response"], epr


def parse_verifier_output(output: str) -> Dict:
    # parse the verifier output to extract JSON fields
    # offers multiple strategies to handle different output formats
    
    original_output = output
    
    # Strategy 1: Try direct JSON parse first
    try:
        result = json.loads(output)
        return {
            "verdict": result.get('verdict', 'N/A'),
            "confidence": result.get('confidence', 'N/A'),
            "error_type": result.get('error_type', result.get('error type', 'N/A')),
            "justification": result.get('justification', 'N/A'),
            "is_reject": result.get('verdict', '') == 'reject',
        }
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract JSON from markdown code blocks
    # Pattern: ```json ... ``` or ``` ... ```
    code_block_pattern = r'```(?:json)?\s*\n(.*?)\n```'
    matches = re.findall(code_block_pattern, output, re.DOTALL)
    
    if matches:
        # Try each match (last one is usually the final answer)
        for match in reversed(matches):
            try:
                result = json.loads(match.strip())
                return {
                    "verdict": result.get('verdict', 'N/A'),
                    "error_type": result.get('error_type', result.get('error type', 'N/A')),
                    "confidence": result.get('confidence', 'N/A'),
                    "justification": result.get('justification', 'N/A'),
                    "is_reject": result.get('verdict', '') == 'reject'
                }
            except json.JSONDecodeError:
                continue
    
    # Strategy 3: Find JSON object anywhere in text
    # Look for { ... } patterns
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    json_matches = re.findall(json_pattern, output, re.DOTALL)
    
    if json_matches:
        # Try from last to first (final answer usually at end)
        for match in reversed(json_matches):
            try:
                result = json.loads(match)
                # Verify it has expected fields
                if 'verdict' in result or 'confidence' in result:
                    return {
                        "verdict": result.get('verdict', 'N/A'),
                        "error_type": result.get('error_type', result.get('error type', 'N/A')),
                        "confidence": result.get('confidence', 'N/A'),
                        "justification": result.get('justification', 'N/A'),
                        "is_reject": result.get('verdict', '') == 'reject'
                    }
            except json.JSONDecodeError:
                continue
    
    # Strategy 4: Try to extract key-value pairs even without proper JSON
    # Look for patterns like "verdict": "accept"
    verdict_match = re.search(r'"verdict"\s*:\s*"(\w+)"', output)
    confidence_match = re.search(r'"confidence"\s*:\s*([\d.]+)', output)
    error_type_match = re.search(r'"error[_ ]type"\s*:\s*"([^"]*)"', output)
    justification_match = re.search(r'"justification"\s*:\s*"([^"]*)"', output)
    
    if verdict_match:
        return {
            "verdict": verdict_match.group(1),
            "error_type": error_type_match.group(1) if error_type_match else "N/A",
            "confidence": confidence_match.group(1) if confidence_match else "N/A",
            "justification": justification_match.group(1) if justification_match else "Partial extraction",
            "is_reject": result.get('verdict', '') == 'reject'

        }
    
    # Failed all strategies
    return {
        "verdict": "N/A",
        "error_type": "Unparseable",
        "confidence": "N/A",
        "justification": f"Could not parse JSON. Output length: {len(output)} chars",
        "is_reject": False
    }


# ============================================================================
# Sampling Functions
# ============================================================================

def load_tn_files(tn_path: Path) -> List[Path]:
    # Load all TN explanation files
    return sorted([f for f in tn_path.glob("*.txt")])


def load_metadata_entries(metadata_path: Path) -> List[Dict]:
    
    if metadata_path.suffix == ".csv":
        
        entries = []
        for _, row in pd.read_csv(metadata_path).iterrows():
            entry = row.to_dict()
            if entry["ground_truth"] == "TN":
                entries.append({
                    "file": entry.get("file", ""),
                    "verdict": entry.get("verifier_verdict", "N/A"),
                    "justification": entry.get("justification", ""),
                    "epr_verifier": entry.get("epr", None),
                    "confidence": entry.get("confidence", None),
                    "error_type": entry.get("error_type", "N/A"),
                })
    
    else:
        # Load all metadata entries from verifier_metadata.txt
        entries = []
        buffer = ""
        with open(metadata_path, "r") as f:                    
            for line in f:     
                buffer += line 
                if line.strip() == "}":
                    try:
                        entries.append(json.loads(buffer))
                    except json.JSONDecodeError:
                        pass
                    finally:
                        buffer = ""
           
    return entries


# match TN file to its metadata entry
def match_file_to_metadata(file_path: Path, metadata_entries: List[Dict]) -> Dict:

    filename = file_path.name
    
    for entry in metadata_entries:
        entry_filename = Path(entry["file"]).name
        if entry_filename == filename:
            return entry


def sample_tn_explanations(tn_path: Path, metadata_path: Path, 
                           n_samples: int, rng: np.random.Generator) -> List[Dict]:
    # Load TN files and metadata
    tn_files = load_tn_files(tn_path)
    metadata_entries = load_metadata_entries(metadata_path)
    
    if len(tn_files) == 0:
        assert False, f"No TN explanation files found in {tn_path}"
    
    # Sample random indices
    n_available = min(n_samples, len(tn_files))
    sampled_indices = rng.choice(len(tn_files), size=n_available, replace=False)
    
    samples = []
    for idx in sampled_indices:
        file_path = tn_files[idx]
        
        # Load explanation text
        with open(file_path, "r") as f:
            explanation_text = f.read()
        
        # Match to metadata
        metadata = match_file_to_metadata(file_path, metadata_entries)
        
        # edge case for deepseek-qwen .csv metadata
        if metadata_path.suffix == ".csv":
            last_line_index = explanation_text.index("EPR Explainer:")
            epr_explainer_value =  float(explanation_text[last_line_index:].split(",")[0].replace("EPR Explainer:", "").strip())
        
        if metadata:
            samples.append({
                "file_path": file_path,
                "explanation_text": explanation_text,
                "justification": metadata.get("justification", ""),
                "epr_explainer": epr_explainer_value if metadata_path.suffix == ".csv" else metadata.get("epr_explainer", None),
                "epr_verifier": metadata.get("epr_verifier", None),
                "confidence": metadata.get("confidence", None),
            })
    
    return samples


def get_mutated_samples(mutation_space: List[Path]): 

    mutated_samples = []

    # For each explanation folder, read the explanations and metadata
    for _, files in mutation_space.items():
        explanations = files["explanations"]
        metadata_files = files["metadata"]
        
        with open(metadata_files[0], "r") as mf:
            metadata_lines = mf.readlines()
        
        for i, expl_file in enumerate(explanations):
            justification = metadata_lines[i].split(":")[1].strip()
            
            with open(expl_file, "r") as ef:
                explanation_text = ef.read()
                
                mutated_samples.append({
                    "file_path": expl_file,
                    "explanation_text": explanation_text,
                    "justification": justification,
                    "epr_explainer": 1.1,           # Declared by us as default values
                    "epr_verifier": 1.0,            # --//--
                    "confidence": 0.97,             # --//--
                })
    return mutated_samples



# ============================================================================
# Feedback Loop Experiment
# ============================================================================

def build_refeed_prompt(xai_output: dict, problematic_explanation: str, 
                       verifier_feedback: str, xai_method: str) -> str:
    
    ref_expl_path = Path("refeed-expl-templates")
    match xai_method:
        case "EBM":
            ref_expl_path /= "ref_expl_template_ebm.md"
        case "Grad-Cam++":
            ref_expl_path /= "ref_expl_template_gradcam.md"
        case "SHAP":
            ref_expl_path /= "ref_expl_template_shap.md"
        case "Integrated Gradients":
            ref_expl_path /= "ref_expl_template_int_grad.md"
        case "LIME":
            ref_expl_path /= "ref_expl_template_lime.md"
        case _:
            assert False, f"Unknown XAI method: {xai_method}"
    
    with open(ref_expl_path, "r") as f:
        template = f.read()
        
        
    template = template.replace("{XAI_CONTENT}", xai_output)
    template = template.replace("{PREV_EXPLANATION}", problematic_explanation)
    template = template.replace("{VER_JUSTIFICATION}", verifier_feedback)
    
    return template


def build_verifier_prompt(xai_output: dict, explanation: Dict[str, str]) -> str:
    
    ref_verif_path = Path("..") / "verifier" / "verifier-templates" / "meta_prompting_template.md"
    
    with open(ref_verif_path, "r") as f:
        template = f.read()
        
    template = template.replace("{XAI_CONTENT}", xai_output)
    template = template.replace("{COT}", explanation["CoT"])
    template = template.replace("{EXPLANATION}", explanation["answer"])
    
    return template


def run_feedback_loop(sample: Dict, arch_config: dict, 
                     max_iterations: int = MAX_REFEED_ITERATIONS) -> Dict:
# Run the feedback loop for a single sample until verified or max iterations reached
    
    # Detect XAI method and load output
    xai_method, xai_output_path = detect_xai_method_from_explanation(
        sample["explanation_text"]
    )
    
    with open(xai_output_path, "r") as f:
        xai_output = json.load(f)
        xai_output_str = json.dumps(xai_output, indent=2)
    
    # Initialize tracking
    epr_explainer_history = [sample["epr_explainer"]]
    epr_verifier_history = [sample["epr_verifier"]]
    current_explanation = sample["explanation_text"]
    verifier_feedback = sample["justification"]
    
    sample_id = sample["file_path"].stem
    
    for iteration in range(max_iterations):
        print(f"  Iteration {iteration + 1}/{max_iterations}...")
        
        # Step 1: Query explainer (with feedback)
        explainer_prompt = build_refeed_prompt(
            xai_output_str, 
            current_explanation, 
            verifier_feedback, 
            xai_method
        )
        
        explainer_output, epr_explainer = query_explainer_stream(arch_config, explainer_prompt)
        epr_explainer_history.append(epr_explainer)
        
        # Extract CoT and answer
        explanation_dict = extract_CoT_answer(explainer_output)
        
        # Step 2: Query verifier
        verifier_prompt = build_verifier_prompt(xai_output_str, explanation_dict)
        verifier_output, epr_ver = query_verifier(arch_config, verifier_prompt)
        epr_verifier_history.append(epr_ver)
        
        # Parse verifier response
        verifier_result = parse_verifier_output(verifier_output)
        verifier_feedback = verifier_result["justification"]
        
        # Check if verified
        if not verifier_result["verdict"] == "reject":
            return {
                "sample_id": sample_id,
                "iterations_to_verify": iteration + 1,
                "final_verdict": "accept",
                "epr_history_explainer": epr_explainer_history,
                "epr_history_verifier": epr_verifier_history,
                "xai_method": xai_method
            }
    
    # Max iterations reached without verification
    return {
        "sample_id": sample_id,
        "iterations_to_verify": max_iterations,
        "final_verdict": "reject",
        "epr_history_explainer": epr_explainer_history,
        "epr_history_verifier": epr_verifier_history,
        "xai_method": xai_method
    }


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment():
    
    config = ExperimentConfig()
    arch_config = load_architecture_config(config.arch_config_path)
    rng = get_numpy_rng(SEED)
    
    all_results = []
    
    print("\n" + "="*60)
    print("FEEDBACK MECHANISM EXPERIMENT")
    print("="*60)
    
    
    mutated_samples = get_mutated_samples(
        config.mutation_space,
    )

    for llm_pair_name, paths in config.llm_pairs.items():
        print(f"\n{'='*60}")
        print(f"Processing LLM Pair: {llm_pair_name}")
        print(f"{'='*60}\n")
        
        # Sample TN explanations
        print(f"Sampling {NATURAL_TN_SAMPLES} TN explanations...")
        natural_sp_samples = sample_tn_explanations(
            paths["tn_path"],
            paths["metadata_path"],
            NATURAL_TN_SAMPLES,
            rng
        )
        
        
        if not natural_sp_samples:
            print(f"⚠ No samples found for {llm_pair_name}. Skipping...")
            continue
        
        print(f"✓ Sampled {len(natural_sp_samples)} explanations from natural space\n")
        
        # Run feedback loop for each sample
        for i, sample in enumerate(natural_sp_samples + mutated_samples):
            print(f"\nSample {i+1}/{len(natural_sp_samples)+len(mutated_samples)}: {sample['file_path'].name}")
            
            result = run_feedback_loop(sample, arch_config)
            result["llm_pair"] = llm_pair_name
            result["Mutated"] = True if i >= len(natural_sp_samples) else False
            all_results.append(result)
            
            # print(f"  ✓ Verified in {result['iterations_to_verify']} iterations")
    
    # Save results
    results_df = pd.DataFrame(all_results)
    output_file = config.output_path / f"feedback_experiment_results_{datetime.datetime.now().strftime('%d-%m-%Y_%H:%M')}.csv"
    results_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Experiment complete! Results saved to: {output_file}")
    print(f"{'='*60}\n")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 60)
    for llm_pair in results_df["llm_pair"].unique():
        pair_data = results_df[results_df["llm_pair"] == llm_pair]
        print(f"\n{llm_pair}:")
        print(f"  Mean iterations: {pair_data['iterations_to_verify'].mean():.2f}")
        print(f"  Median iterations: {pair_data['iterations_to_verify'].median():.0f}")
        print(f"  Verified: {(pair_data['final_verdict'] == 'accept').sum()}/{len(pair_data)}")
    
    return results_df


if __name__ == "__main__":
    start_time = pcnt()
    results = run_experiment()
    end_time = pcnt()
    total_time = end_time - start_time
    print(f"Total experiment time: {total_time/60:.2f} minutes")