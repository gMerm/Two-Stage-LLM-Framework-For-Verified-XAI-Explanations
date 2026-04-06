"""
This script runs once for each dataset & paired XAI method found in the XAI-Methods-outputs folder.
"""
import os
import json
import argparse
import yaml
import requests
from pathlib import Path
from tqdm import tqdm
from io import StringIO
from typing import Dict


def load_config(config_path='../architecture.yaml'):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def query_explainer(config, prompt):
    print(f"2. Querying explainer model at {config['host']}:{config['port']} with model {config['Explainer']['model']}")
    url = f"{config['host']}:{config['port']}/api/generate"
    
    # https://ollama.com/blog/thinking
    payload = {
        "model": config['Explainer']["model"],
        "prompt": prompt,
        "temperature": config['Explainer'].get("temperature", 0.7),
        "top_p": config['Explainer'].get("top_p", 0.9),
        "stream": True,
        "think": True       # new ollama version needs this to enable thinking stream
    }
    
    response = requests.post(url, json=payload, stream=True)
    response.raise_for_status()
    
    # new buffers for thinking and answer streams
    thinking_buffer = StringIO()
    answer_buffer = StringIO()
    
    with tqdm(unit=f"chunks", desc="Receiving LLM answer stream") as pbar:
        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                    continue
            
            # The new 'thinking' stream
            if thinking_text := data.get("thinking", ""):
                thinking_buffer.write(thinking_text)
            
            # The old 'response' stream (now only the final answer)
            if answer_text := data.get("response", ""):
                answer_buffer.write(answer_text)
            
            pbar.update(1) 
    
    
    # Combine thinking and answer into final output
    full_output = (
        f"<think>{thinking_buffer.getvalue()}</think>" + '\n' +
        f"{answer_buffer.getvalue()}"
    )
    
    return full_output

def extract_CoT_answer(full_output: str) -> Dict[str,str]:
    start_CoT_tag = f"<think>"
    end_CoT_tag = f"</think>"
    
    if start_CoT_tag in full_output and end_CoT_tag in full_output:
        start_ans_pos = full_output.find(start_CoT_tag) + len(start_CoT_tag)
        end_ans_pos = full_output.find(end_CoT_tag)
        CoT = full_output[start_ans_pos:end_ans_pos].strip()
        answer = full_output[end_ans_pos + len(end_CoT_tag):].strip() 
    
        return {
            f"CoT": CoT,
            f"answer": answer,
            "raw_expl_output": full_output
        }
    else:
        return {
            f"CoT": "",
            f"answer": full_output,
            "raw_expl_output": full_output
        }

def build_prompt(template_path, xai_output_path):
    print(f"1. Building prompt from template: {template_path}")
    with open(template_path, "r") as f:
        template = f.read()
    with open(xai_output_path, "r") as f:
        xai_data = json.load(f)

    template = template + '\n' + json.dumps(xai_data, indent=2) + '\n'
    return template

def run_case(config, dataset, method, xai_file, template_file, out_path):
    
    print(f"Running case: Dataset={dataset}, Method={method}, XAI File={xai_file}")
    prompt = build_prompt(template_file, xai_file)
    output = query_explainer(config, prompt)
    
    out_json = extract_CoT_answer(output)

    # import code; code.interact(local=locals())
    with open(out_path, "w", encoding="utf-8") as f, open(f"{str(out_path)[:-4]}txt", "w", encoding="utf-8") as txt_f:
        json.dump(out_json, f, ensure_ascii=False, indent=4)
        txt_f.write(out_json["raw_expl_output"])

    print(f"3. Saved explanation to {out_path} and {str(out_path)[:-4]}txt\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run all cases")
    parser.add_argument("-d", "--dataset", type=str, help="Run single dataset")
    args = parser.parse_args()
    
    # Validate args
    if not args.all and (not args.dataset):
        parser.error("Must specify --all or --dataset")

    # Load config
    config = load_config(os.path.join(f"..", f"architecture.yaml"))
    
    xai_outputs_folder = Path("../../XAI-Methods-outputs")
    templates_folder = Path("explainer-templates")
    results_folder = Path("./results")
    results_folder.mkdir(exist_ok=True)
    
    # Define dataset configurations
    dataset_config = {
        "ACS_Income": {"template": "shap_template.md", "method": "shap"},
        "CIFAR10": {"template": "grad_cam_template.md", "method": "grad_campp"},
        "IMDB": {"template": "integrated_gradients_template.md", "method": "integrated_gradients"},
        "Diamonds": {"template": "diamonds.LIME.template.md", "method": "LIME"},
        "Wine": {"template": "wine.EBM.template.md", "method": "EBM"},
    }
    
    # Firstly determine datasets to process (based on user flag)
    if args.all:
        datasets_to_process = [d for d in xai_outputs_folder.iterdir() if d.is_dir()]
    else:
        dataset_path = xai_outputs_folder / args.dataset
        if not dataset_path.exists():
            print(f"Error: Dataset folder '{args.dataset}' not found")
            exit(1)
        datasets_to_process = [dataset_path]
    
    # Now process each dataset
    for dataset_dir in datasets_to_process:
        dataset = dataset_dir.name
        
        if dataset not in dataset_config:
            print(f"Unknown dataset {dataset}, skipping...\n")
            continue
            
        template_file = templates_folder / dataset_config[dataset]["template"]
        method = dataset_config[dataset]["method"]
        
        for xai_result_file in dataset_dir.glob("*.json"):
            output_path = results_folder / f"{xai_result_file.stem}_explanation.json"
            run_case(config, dataset, method, xai_result_file, template_file, output_path)
    
# run main
if __name__ == "__main__":
    main()