"""
This script is designed to test a verifier model on already categorized explanations.
So before running this script, run "generate_errors_ollama.py" to create explanations,
manually categorize them into TP, FP, TN, FN folders, and then run this script.
"""

import json
import yaml
import requests
import os
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from generate_errors_ollama import calculate_EPR
import re

@dataclass
class VerifierTestConfig:
    arch_config: dict
    verifier_template_path: Path
    processed_results_path: Path
    output_path: Path
    xai_method_outputs: Path
    
    def __post_init__(self):
        self.output_path.mkdir(parents=True, exist_ok=True)
        print(f"Output directory created/verified: {self.output_path}")


def load_config(config_path='../architecture.yaml'):
    print(f"Loading configuration from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"Configuration loaded successfully")
    return config


def query_verifier(config, prompt):
    # Query the verifier model
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


def extract_explanation_from_file(file_path):
    # Extract the explanation and CoT from a saved file
    with open(file_path, "r") as f:
        content = f.read()
    
    # Try to extract CoT and answer
    cot = ""
    answer = content
    
    if "<think>" in content and "</think>" in content:
        start = content.find("<think>") + len("<think>")
        end = content.find("</think>")
        cot = content[start:end].strip()
        answer = content[end + len("</think>"):].strip()
    
    return {"CoT": cot, "answer": answer, "raw_content": content}


def detect_xai_method_from_explanation(explanation_text):
    # Detect which XAI method was used based on keywords in the explanation
    # Returns: (xai_method_name, xai_output_path)
    
    # Check for keywords in the explanation
    if "Sulphates" in explanation_text or "sulphates" in explanation_text:
        return "EBM", Path("Wine") / "wine.EBM.out.json"
    elif "Middle-Center" in explanation_text or "middle-center" in explanation_text:
        return "Grad-Cam++", Path("CIFAR10") / "gradcampp_sample2_output.json"
    elif "WKHP" in explanation_text:
        return "SHAP", Path("ACS_Income") / "shap_acs_output.json"
    elif "films" in explanation_text or "film" in explanation_text:
        return "Integrated Gradients", Path("IMDB") / "int_gradients_sample198_output.json"
    elif "Carat" in explanation_text or "carat" in explanation_text:
        return "LIME", Path("Diamonds") / "LIME_diamonds.out.json"
    else:
        # Default fallback - could also raise an error
        print(f"Warning: Could not detect XAI method from explanation. Using SHAP as default.")
        return "Unknown", Path("ACS_Income") / "shap_acs_output.json"


def build_verifier_prompt_from_explanation(template_path, explanation_dict, xai_method_outputs_base_path):
    # Build verifier prompt from extracted explanation with proper XAI content
    print(f"  Loading template from: {template_path}")
    with open(template_path, "r") as f:
        template = f.read()
    
    # Detect which XAI method was used
    full_text = explanation_dict.get("raw_content", "")
    xai_method, xai_output_relative_path = detect_xai_method_from_explanation(full_text)
    print(f"  Detected XAI method: {xai_method}")
    
    # Load the corresponding XAI method output
    xai_output_path = xai_method_outputs_base_path / xai_output_relative_path
    print(f"  Loading XAI output from: {xai_output_path}")
    
    try:
        with open(xai_output_path, "r") as f:
            xai_method_output = json.load(f)
        xai_content_str = json.dumps(xai_method_output, indent=2)
        print(f"  XAI output loaded successfully")
    except FileNotFoundError:
        print(f"  Warning: XAI output file not found: {xai_output_path}")
        xai_content_str = "N/A - XAI output file not found"
    except json.JSONDecodeError:
        print(f"  Warning: Could not parse XAI output: {xai_output_path}")
        xai_content_str = "N/A - Invalid JSON"
    
    # Build prompt with actual XAI content
    prompt = template.replace("{XAI_CONTENT}", xai_content_str)
    prompt = prompt.replace("{COT}", explanation_dict.get("CoT", ""))
    prompt = prompt.replace("{EXPLANATION}", explanation_dict.get("answer", ""))
    
    print(f"  Prompt built successfully (length: {len(prompt)} chars)")
    return prompt, xai_method


def process_verifier_output(output):
    # parse the verifier output to extract JSON fields
    # offers multiple strategies to handle different output formats
    
    original_output = output
    
    # Strategy 1: Try direct JSON parse first
    try:
        result = json.loads(output)
        return {
            "verdict": result.get('verdict', 'N/A'),
            "error_type": result.get('error_type', result.get('error type', 'N/A')),
            "confidence": result.get('confidence', 'N/A'),
            "justification": result.get('justification', 'N/A'),
            "parseable": True,
            "extraction_method": "direct_json"
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
                    "parseable": True,
                    "extraction_method": "code_block"
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
                        "parseable": True,
                        "extraction_method": "json_search"
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
            "parseable": True
        }
    
    # Failed all strategies
    return {
        "verdict": "N/A",
        "error_type": "Unparseable",
        "confidence": "N/A",
        "justification": f"Could not parse JSON. Output length: {len(output)} chars",
        "parseable": False,
    }



def map_verdict_to_category(ground_truth_category, verdict):
    """
    Map verifier verdict to predicted category.
    
    Ground truth categories (manual labels):
    - TP: Good explanation that WAS accepted (should accept)
    - TN: Bad explanation that WAS rejected (should reject)
    - FP: Bad explanation that WAS accepted (should reject)
    - FN: Good explanation that WAS rejected (should accept)
    """
    
    # What should happen to this explanation?
    should_accept = ground_truth_category in ["TP", "FN"]  # Good explanations
    should_reject = ground_truth_category in ["TN", "FP"]  # Bad explanations
    
    # What did the verifier do?
    verifier_accepts = verdict == "accept"
    verifier_rejects = verdict == "reject"
    
    # Determine the resulting category
    if should_accept and verifier_accepts:
        return "TP" 
    elif should_accept and verifier_rejects:
        return "FN" 
    elif should_reject and verifier_rejects:
        return "TN" 
    elif should_reject and verifier_accepts:
        return "FP"
    else:
        return "UNKNOWN"


def test_verifier_on_categorized_data(config):
    # Main function to test verifier on manually categorized data
    categories = ["TP", "FP", "TN", "FN"]
    results = []
    
    # create output directory for raw verifier outputs
    raw_outputs_dir = config.output_path / "raw_verifier_outputs"
    raw_outputs_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Testing Verifier: {config.arch_config['Verifier']['model']}")
    print(f"{'='*60}\n")
    
    # Process each category folder
    for ground_truth_cat in categories:
        cat_folder = config.processed_results_path / ground_truth_cat
        
        if not cat_folder.exists():
            print(f"Warning: Folder {cat_folder} does not exist. Skipping...")
            continue
        
        # Get all .txt files in this category
        txt_files = list(cat_folder.glob("*.txt"))
        print(f"\n{'='*60}")
        print(f"Processing {ground_truth_cat} category: {len(txt_files)} files")
        print(f"Folder: {cat_folder}")
        print(f"{'='*60}")
        
        for idx, file_path in enumerate(tqdm(txt_files, desc=f"Testing {ground_truth_cat}"), 1):
            print(f"\n[{idx}/{len(txt_files)}] Processing: {file_path.name}")
            
            # Extract explanation
            print(f"  Extracting explanation from file...")
            explanation = extract_explanation_from_file(file_path)
            print(f"  Explanation extracted (CoT: {len(explanation['CoT'])} chars, Answer: {len(explanation['answer'])} chars)")
            
            # Build verifier prompt
            try:
                prompt, xai_method = build_verifier_prompt_from_explanation(
                    config.verifier_template_path, 
                    explanation,
                    config.xai_method_outputs
                )
            except Exception as e:
                print(f"  Error building prompt: {e}")
                continue
            
            # Query verifier
            print(f"  Querying verifier model...")
            try:
                verifier_output, epr = query_verifier(config.arch_config, prompt)
                
                # Save raw verifier output
                raw_output_file = raw_outputs_dir / f"{ground_truth_cat}_{file_path.stem}_verifier_output.txt"
                with open(raw_output_file, "w") as f:
                    f.write(verifier_output)
                    
                print(f"  Verifier response received (length: {len(verifier_output)} chars)")
                parsed = process_verifier_output(verifier_output)
                parsed["epr"] = epr
                print(f"  Verdict: {parsed['verdict']}, Error Type: {parsed['error_type']}, EPR: {epr:.4f}")
            except Exception as e:
                print(f"  Error querying verifier: {e}")
                parsed = {
                    "verdict": "ERROR",
                    "error_type": "API_ERROR",
                    "confidence": "N/A",
                    "justification": str(e),
                    "parseable": False,
                    "epr": 0.0
                }
            
            # Determine predicted category
            predicted_cat = map_verdict_to_category(ground_truth_cat, parsed["verdict"])
            match_status = "MATCH" if predicted_cat == ground_truth_cat else "MISMATCH"
            print(f"  Ground Truth: {ground_truth_cat} -> Predicted: {predicted_cat} [{match_status}]")
            
            # Store results
            results.append({
                "file": file_path.name,
                "ground_truth": ground_truth_cat,
                "verifier_verdict": parsed["verdict"],
                "predicted_category": predicted_cat,
                "error_type": parsed["error_type"],
                "confidence": parsed["confidence"],
                "justification": parsed["justification"],
                "parseable": parsed["parseable"],
                "xai_method": xai_method,
                "epr": parsed["epr"]
            })
    
    print(f"\n{'='*60}")
    print(f"Processed {len(results)} total files")
    print(f"{'='*60}\n")
    
    return results


def generate_analysis(results, config):
    # Generate analysis and confusion matrix
    print(f"\n{'='*60}")
    print("GENERATING ANALYSIS")
    print(f"{'='*60}\n")
    
    df = pd.DataFrame(results)
    
    # Save detailed results
    output_file = config.output_path / "verifier_test_results.csv"
    df.to_csv(output_file, index=False)
    print(f"Detailed results saved to: {output_file}")
    
    # Generate confusion matrix
    categories = ["TP", "FP", "TN", "FN"]
    confusion = pd.crosstab(
        df['ground_truth'], 
        df['predicted_category'],
        rownames=['Ground Truth'],
        colnames=['Predicted'],
        dropna=False
    )
    
    # Ensure all categories are present
    for cat in categories:
        if cat not in confusion.index:
            confusion.loc[cat] = 0
        if cat not in confusion.columns:
            confusion[cat] = 0
    
    confusion = confusion.reindex(index=categories, columns=categories, fill_value=0)
    
    # Save confusion matrix
    confusion_file = config.output_path / "confusion_matrix.csv"
    confusion.to_csv(confusion_file)
    print(f"Confusion matrix saved to: {confusion_file}")
    
    # Calculate metrics
    print(f"\n{'='*60}")
    print("CONFUSION MATRIX")
    print(f"{'='*60}")
    print(confusion)
    
    # Calculate accuracy
    correct = sum([confusion.loc[cat, cat] for cat in categories])
    total = df.shape[0]
    accuracy = correct / total if total > 0 else 0
    
    # Agreement with ground truth
    agreement = (df['ground_truth'] == df['predicted_category']).sum() / total if total > 0 else 0
    
    print(f"\n{'='*60}")
    print("METRICS")
    print(f"{'='*60}")
    print(f"Total samples: {total}")
    print(f"Agreement with manual categorization: {agreement:.2%}")
    print(f"Parseable responses: {df['parseable'].sum()}/{total}")
    
    # Error type distribution
    print(f"\n{'='*60}")
    print("ERROR TYPE DISTRIBUTION")
    print(f"{'='*60}")
    error_type_dist = df['error_type'].value_counts()
    print(error_type_dist)
    
    # XAI method distribution
    print(f"\n{'='*60}")
    print("XAI METHOD DISTRIBUTION")
    print(f"{'='*60}")
    xai_method_dist = df['xai_method'].value_counts()
    print(xai_method_dist)
    
    # Save summary
    summary = {
        "verifier_model": config.arch_config['Verifier']['model'],
        "total_samples": int(total),
        "agreement_rate": float(agreement),
        "parseable_count": int(df['parseable'].sum()),
        "epr": float(df['epr'].mean()),
        "confusion_matrix": confusion.to_dict('index'),
        "error_type_distribution": error_type_dist.to_dict(),
        "xai_method_distribution": xai_method_dist.to_dict()
    }
    
    summary_file = config.output_path / "test_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")
    
    return df, confusion


def main():
    print(f"\n{'='*60}")
    print("VERIFIER TESTING SCRIPT")
    print(f"{'='*60}\n")
    
    # Setup configuration
    print("Setting up configuration...")
    config = VerifierTestConfig(
        arch_config=load_config(os.path.join("..", "architecture.yaml")),
        verifier_template_path=Path("verifier-templates") / "meta_prompting_template.md",
        processed_results_path=Path("..") / "explainer" / "results" / "explanations_space_deep_gpt" / "processed_results",
        output_path=Path("..") / "explainer" / "results" / "verifier_test_results_deep_qwen",
        xai_method_outputs=Path("../..") / "XAI-Methods-outputs"
    )
    
    print(f"\nConfiguration Summary:")
    print(f"  - Verifier model: {config.arch_config['Verifier']['model']}")
    print(f"  - Template: {config.verifier_template_path}")
    print(f"  - Input folder: {config.processed_results_path}")
    print(f"  - Output folder: {config.output_path}")
    print(f"  - XAI outputs: {config.xai_method_outputs}")
    
    # Test verifier
    print(f"\nStarting verifier testing...")
    results = test_verifier_on_categorized_data(config)
    
    # Generate analysis
    if results:
        generate_analysis(results, config)
        print(f"\n{'='*60}")
        print("TESTING COMPLETED SUCCESSFULLY")
        print(f"{'='*60}\n")
    else:
        print("\nWarning: No results generated. Check your data paths.")


if __name__ == "__main__":
    main()