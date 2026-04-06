import json
import yaml
import requests
import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from io import StringIO
from tqdm import tqdm
from time import perf_counter as timer
import numpy as np

VERBOSE = False

@dataclass
class Configuration:
    # LLM and file paths parameters
    arch_config: dict
    xai_method_outputs: Path
    explainer_templates_path: Path  
    verifier_template_path: Path

    # simulation parameters
    XAI_METHODS: tuple[str, ...] = ("SHAP", "LIME", "EBM", "Integrated Gradients", "Grad-Cam++")
    errors_found: int = 0
    target_errors: int = 200
    max_attempts: int = 1000
    max_omit_errors: int = 40
    
    # output paths
    explanations_folder: Path = Path("..") / "explainer" / "results" / "explanations_space_1"

    # post init folders
    def __post_init__(self):
        self.explanations_folder.mkdir(parents=True, exist_ok=True)
        (self.explanations_folder / f"accepted").mkdir(parents=True, exist_ok=True)
        (self.explanations_folder / f"rejected").mkdir(parents=True, exist_ok=True)
        
        self.metadata_path = self.explanations_folder / "verifier_metadata.txt"
        if not self.metadata_path.exists():
            with open(self.metadata_path, "w") as f:
                json.dump({}, f)


def load_config(config_path='../architecture.yaml'):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def calculate_EPR(logprobs_info) -> float:
    if not logprobs_info or not isinstance(logprobs_info, list):
        return 0.0
    
    Lq = len(logprobs_info)
    log_probs_matrix = np.full((Lq, 10), np.nan, dtype=np.float32)

    for i, logprob_obj in enumerate(logprobs_info):
        if logprob_obj and "top_logprobs" in logprob_obj and logprob_obj["top_logprobs"]:
            # Extract logprob values from the list of dicts
            vals = [item["logprob"] for item in logprob_obj["top_logprobs"]]
            n = min(len(vals), 10)
            log_probs_matrix[i, :n] = vals[:n]
    
    if np.all(np.isnan(log_probs_matrix)):
        return 0.0  # No valid logprobs found
        
    # Actual EPR calculation
    log_probs = np.atleast_2d(log_probs_matrix)
    lp_max = np.nanmax(log_probs, axis=1, keepdims=True)
    probs = np.exp(log_probs - lp_max)
    
    # Handle NaN values in probability calculation
    probs = np.nan_to_num(probs, nan=0.0)
    prob_sums = np.sum(probs, axis=1, keepdims=True)
    prob_sums[prob_sums == 0] = 1.0  # Avoid division by zero
    probs /= prob_sums
    
    # Calculate entropy
    entropies = -np.sum(probs * np.log2(probs + 1e-10), axis=1)
    epr = float(np.mean(entropies))
    
    if np.isnan(epr):
        return 0.0

    return epr


def query_verifier(config, prompt):
    url = f"{config['host']}:{config['port']}/api/generate"
    payload = {
        "model": config["Verifier"]["model"],
        "prompt": prompt,
        "temperature": config["Verifier"].get("temperature", 0.6),
        "stream": False,
        "think": True if config['Verifier']["model"] not in ["phi4-reasoning:latest"] else False,
        "logprobs": True,
        "top_logprobs": 10,
        "options": {
            "num_ctx": config["Verifier"].get("num_ctx", 131072)    # 128k context
        }
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    result = response.json()
    
    logprobs = result.get("logprobs", [])
    epr = calculate_EPR(logprobs) if logprobs else 0.0
    
    return result["response"], epr


def query_explainer_stream(config, prompt):

    print(f"Querying explainer model: {config['Explainer']['model']}")
    
    url = f"{config['host']}:{config['port']}/api/generate"
    payload = {
        "model": config["Explainer"]["model"],
        "prompt": prompt,
        "temperature": config["Explainer"].get("temperature", 2.5),
        "top_p": config["Explainer"].get("top_p", 1.0),
        "top_k": config["Explainer"].get("top_k", 0),
        "seed": config["Explainer"].get("seed", -1),
        "stream": True,
        # model dependent - Check new ollama docs for "think" param
        "think": True if config['Explainer']["model"] not in ["phi4-reasoning:latest"] else False,
        "logprobs": True,
        "top_logprobs": 10
    }
    
    response = requests.post(url, json=payload, stream=True)
    response.raise_for_status()
    
    thinking_buffer = StringIO()
    answer_buffer = StringIO()
    all_logprobs = []

    with tqdm(unit="chunks", desc="Generating explanation") as pbar:
        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            if "response" not in data:
                continue

            if thinking_text := data.get("thinking", ""):
                thinking_buffer.write(thinking_text)
                
            if answer_text := data.get("response", ""):
                answer_buffer.write(answer_text)
                
            # Collect logprobs if available
            if "logprobs" in data and data["logprobs"]:
                all_logprobs.extend(data["logprobs"])
                
            pbar.update(1)
    
    full_output = (
        f"<think>{thinking_buffer.getvalue()}</think>" + '\n' +
        f"{answer_buffer.getvalue()}"
    )
    
    # calculate epr
    epr = calculate_EPR(all_logprobs)

    return full_output, epr


def extract_CoT_answer(full_output: str):
    start_CoT_tag = f"<think>"
    end_CoT_tag = f"</think>"
    CoT = f""
    answer = full_output
    
    if start_CoT_tag in full_output and end_CoT_tag in full_output:
        start_ans_pos = full_output.find(start_CoT_tag) + len(start_CoT_tag)
        end_ans_pos = full_output.find(end_CoT_tag)
        CoT = full_output[start_ans_pos:end_ans_pos].strip()
        answer = full_output[end_ans_pos + len(end_CoT_tag):].strip()
        
    return {
        "CoT": CoT,
        "answer": answer,
        "raw_expl_output": full_output
        }


def sanitize_filename(text):
    text = text.replace('/', '_').replace('\\', '_')
    text = re.sub(r'[<>:"|?*]', '_', text)
    
    return text[:100] # Limit length to 100 characters


def explainer_generate(explainer_template_path, xai_method_output, arch_config):
    
    # Load XAI method template
    with open(explainer_template_path, "r") as template:
        xai_template = template.read()
    
    # Build prompt
    xai_content_str = json.dumps(xai_method_output, indent=2)
    prompt = xai_template.replace("{XAI_CONTENT}", xai_content_str)

    # Query model
    output, epr = query_explainer_stream(arch_config, prompt)
    
    # Extract CoT and answer, also epr
    output_json = extract_CoT_answer(output)
    output_json["epr"] = epr
    
    return output_json  


def build_verifier_prompt(template_path, xai_method_output, generated_explanation):
    
    # Load verifier template
    with open(template_path, "r") as f:
        template = f.read()

    # Replace placeholders in template
    xai_method_output_str = json.dumps(xai_method_output, indent=2)
    prompt = template.replace("{XAI_CONTENT}", xai_method_output_str)
    prompt = prompt.replace("{COT}", generated_explanation.get("CoT", ""))
    prompt = prompt.replace("{EXPLANATION}", generated_explanation.get("answer", ""))
    
    return prompt


def process_verifier_output(output, epr):
    try: 
        # Parsing as JSON according to the Verifier schema
        result = json.loads(output)
        verdict = result.get('verdict', 'N/A')
        confidence = result.get('confidence', 'N/A')
        justification = result.get('justification', 'N/A')
        error_type = result.get('error_type', result.get('error type', 'N/A'))
        
        if VERBOSE:
            print(f"\n- Verdict: {verdict}")
            print(f"- Confidence: {confidence}")
            print(f"- Error Type: {error_type}")
            print(f"- Justification: {justification}")
            print(f"- EPR: {epr}\n")

        return verdict == "reject", sanitize_filename(str(error_type)), justification, confidence, epr

    except json.JSONDecodeError:
        print("\n⚠ Warning: Could not parse verifier output as JSON")
        return False, "Unparseable", "Unparseable", "Unparseable", epr


def log_verifier_response(metadata_path, explanations_folder, error_type, verdict, justification, 
                            confidence, explainer_output, error_index, attempt, expl_epr, ver_epr):

    if not verdict:
        accepted_folder = explanations_folder / "accepted"
        accepted_file_name = f"accepted_{attempt}.txt"
        
        with open(accepted_folder / accepted_file_name, "w") as f:
            f.write(explainer_output.get("raw_expl_output", ""))
            f.write(f"\n\n# EPR Explainer: {expl_epr:.4f}, Verifier: {ver_epr:.4f}\n")
    
    else:
        errors_folder = explanations_folder / "rejected"
        error_file_name = f"error_{error_index}_{error_type}.txt"
        error_path = errors_folder / error_file_name
        
        error_entry = {
            "error_number": error_index,
            "attempt": attempt,
            "verdict": "reject" if verdict else "accept",
            "confidence": confidence,
            "error_type": error_type,
            "justification": justification,
            "epr_explainer": expl_epr,
            "epr_verifier": ver_epr,
            "file": str(error_path)
        }

        with open(error_path, "w") as f, open(metadata_path, "a") as metadata_file:
            f.write(explainer_output.get("raw_expl_output", ""))
            f.write(f"\n\n# EPR Explainer: {expl_epr:.4f}, Verifier: {ver_epr:.4f}\n")
            json.dump(error_entry, metadata_file, indent=4, ensure_ascii=False)
            metadata_file.write('\n')

        print(f"ERROR DETECTED! Saved to: {error_path}")


def main():
    
    # Load configs
    config = Configuration(
        arch_config=load_config(os.path.join(f"..", f"architecture.yaml")),
        explainer_templates_path = Path("..") / "explainer" / "explainer-templates",
        xai_method_outputs=Path("..") / f".." / "XAI-Methods-outputs",
        verifier_template_path=Path("verifier-templates") / f"meta_prompting_template.md",
    )

    print(f"\n{'='*60}")
    print(f"Starting Pipeline loop: Looking for {config.target_errors} errors")
    print(f"{'='*60}\n")
    omit_errors = 0
    
    # Main pipeline loop
    tstart = timer()
    for attempt in range(1, config.max_attempts + 1):
            # Check if target errors reached
            if config.errors_found >= config.target_errors:
                print(f"\nReached target of {config.target_errors} errors. Stopping...")
                break
            
            # Select XAI method RR fashion
            xai_method = config.XAI_METHODS[attempt % len(config.XAI_METHODS)]
            
            match xai_method:
                case "SHAP":
                    xai_template_path = config.explainer_templates_path / f"shap_template.md"
                    xai_output_path = config.xai_method_outputs / f"ACS_Income" / f"shap_acs_output.json"
                case "LIME":
                    xai_template_path = config.explainer_templates_path / f"lime_template.md"
                    xai_output_path = config.xai_method_outputs / f"Diamonds" / f"LIME_diamonds.out.json" 
                case "EBM":
                    xai_template_path = config.explainer_templates_path / f"ebm_template.md"
                    xai_output_path = config.xai_method_outputs / f"Wine" / f"wine.EBM.out.json" 
                case "Integrated Gradients":
                    xai_template_path = config.explainer_templates_path / f"integrated_gradients_template.md"
                    xai_output_path = config.xai_method_outputs / f"IMDB" / f"int_gradients_sample198_output.json"
                case "Grad-Cam++":
                    xai_template_path = config.explainer_templates_path / f"grad_cam_template.md"
                    xai_output_path = config.xai_method_outputs / f"CIFAR10" / f"gradcampp_sample2_output.json"         
                case _:
                    raise ValueError(f"Unknown XAI method: {xai_method}")
            
            # Load output only once
            print(f"Loading XAI output from: {xai_output_path}")
            with open(xai_output_path, "r") as f_output:
                xai_method_output = json.load(f_output)
            
            print("[1/3] Running Explainer...")
            try:
                generated_explanation = explainer_generate(xai_template_path, xai_method_output, config.arch_config)
            except Exception as e:
                print(f"Error in Explainer generation: {e}")
                continue
            
            # Build verifier prompt
            print("[2/3] Building verifier prompt...")
            verifier_prompt = build_verifier_prompt(config.verifier_template_path, xai_method_output, generated_explanation)
            
            # Query verifier
            print("[3/3] Verifying explanation...")
            verifier_output, ver_epr = query_verifier(config.arch_config, verifier_prompt)
            verdict, error_type, justification, confidence, ver_epr = process_verifier_output(verifier_output, ver_epr)
            if omit_errors >= config.max_omit_errors and error_type == "OmitFeature":
                continue
            if error_type == "OmitFeature":
                omit_errors += 1
            
            if error_type == "Unparseable":
                with open(config.metadata_path, "a") as f:
                    f.write(f"\nAttempt {attempt}: Unparseable verifier output. Skipping...\n")
                continue
            
            config.errors_found += int(verdict)
            log_verifier_response(config.metadata_path, config.explanations_folder, error_type, verdict, 
                                justification, confidence, generated_explanation, error_index=config.errors_found, attempt=attempt,
                                expl_epr=generated_explanation.get("epr", 0.0), ver_epr=ver_epr)
    
    tend = timer()
    
    with open(config.metadata_path, "a") as f:
        f.write(f"\nTotal time: {tend - tstart:.2f} seconds\n")
        f.write(f"GENERATION COMPLETED! Found {config.errors_found} errors in {attempt} attempts\n")
        f.write(f"Omit errors: {omit_errors}\n")
        
    print(f"\n{'='*60}")
    print(f"Total time: {tend - tstart:.2f} seconds")
    print(f"GENERATION COMPLETED! Found {config.errors_found} errors in {attempt} attempts")
    print(f"Results saved in: {config.explanations_folder}")
    print(f"Error log: {config.metadata_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()