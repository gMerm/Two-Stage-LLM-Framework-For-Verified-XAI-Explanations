"""
This script runs the Verifir once for the specified explanation file.
"""
import json
import yaml
import requests
import os
from pathlib import Path

def load_config(config_path='../architecture.yaml'):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def query_verifier(config, prompt):
    print(f"Querying verifier model at {config['host']}:{config['port']} with model {config['Verifier']['model']}")
    url = f"{config['host']}:{config['port']}/api/generate"
    payload = {
        "model": config["Verifier"]["model"],
        "prompt": prompt,
        "temperature": config['Verifier'].get("temperature", 0.6),
        "stream": False,
        "think": True if config['Verifier']["model"] not in ["phi4-reasoning:latest"] else False,
        #"think": False
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    return response.json()["response"]

def build_verifier_prompt(template_path, xai_content_path, explainer_output_path):
    print(f"Building verifier prompt...")
    
    # Load template
    with open(template_path, "r") as f:
        template = f.read()
    
    # Load XAI content
    with open(xai_content_path, "r") as f:
        xai_data = json.load(f)
    
    # Load explainer output
    with open(explainer_output_path, "r") as f:
        explainer_data = json.load(f)
    
    # Replace placeholders in template
    prompt = template.replace("{XAI_CONTENT}", json.dumps(xai_data, indent=2))
    prompt = prompt.replace("{COT}", explainer_data.get("CoT", ""))
    prompt = prompt.replace("{EXPLANATION}", explainer_data.get("answer", ""))
    
    return prompt

def main():
    # Load config
    config = load_config(os.path.join(f"..", f"architecture.yaml"))
    
    for mut in range(1,7,1):
        print(f"\n--- Running verifier for mutation {mut} ---")
        
        # Define paths
        template_path = Path("verifier-templates/meta_prompting_template.md")
        #xai_content_path = Path(f"../../XAI-Methods-outputs/Wine/wine.EBM.out.json")
        #explainer_output_path = Path(f"../explainer/results/wine_mutated/wine.EBM.out_explanation_mut{mut}.json")
        xai_content_path = Path(f"../../XAI-Methods-outputs/Diamonds/LIME_diamonds.out.json")
        explainer_output_path = Path(f"../explainer/results/diamonds_mutated/LIME_diamonds.out_explanation_mut{mut}.json")
        
        print(f"\n=== VERIFIER PROOF OF CONCEPT ===")
        print(f"XAI Content: {xai_content_path}")
        print(f"Explainer Output: {explainer_output_path}")
        print(f"Template: {template_path}\n")
        
        # Build prompt
        prompt = build_verifier_prompt(template_path, xai_content_path, explainer_output_path)
        
        # Query verifier
        print("Sending to verifier model...")
        output = query_verifier(config, prompt)
        
        print("\n=== VERIFIER OUTPUT ===")
        print(output)
        print("\n======================")
        
        # Try to parse as JSON
        try:
            result = json.loads(output)
            print("\n=== PARSED RESULT ===")
            print(f"Verdict: {result.get('verdict', 'N/A')}")
            print(f"Confidence: {result.get('confidence', 'N/A')}")
            print(f"Error Type: {result.get('error_type', 'N/A')}")
            print(f"Justification: {result.get('justification', 'N/A')}")
        except json.JSONDecodeError:
            print("\nWarning: Could not parse output as JSON")
        
        # Save output
        model = config['Verifier']['model'].replace(":", "_").replace("/", "_")
        #output_path = Path(f"results/verifier_output_mut_EBMs_{model}.txt")
        output_path = Path(f"results/verifier_output_mut_LIME_{model}.txt")
        with open(output_path, "a") as f:
            f.write(output)
            f.write("\n\n")
        print(f"\nSaved raw output to: {output_path}")

if __name__ == "__main__":
    main()