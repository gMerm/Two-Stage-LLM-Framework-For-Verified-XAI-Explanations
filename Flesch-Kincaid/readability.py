"""
1. Baseline: Raw XAI method outputs straight from folder "XAI-Methods-outputs"
2. For the LLM explanations:
    a. Scan TP, TN folders
    b. For each .txt file found in those folders:
        i.      Clean the text removing <think> tags, EPR footer & XML tags
        ii.     Detect XAI method used based on keywords
        iii.    Calculate readability metrics on cleaned text
        iv.     Aggregate results per XAI method
        v.      Average the readability metrics across all 5 XAI methods
        
- Flesh: Higher score   = easier to read
- Grade: Lower score    = easier to read
"""

import json
import re
from pathlib import Path
import textstat
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def extract_text_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    text_parts = []
    
    def extract_strings(obj):
        if isinstance(obj, str):
            text_parts.append(obj)
        elif isinstance(obj, dict):
            for value in obj.values():
                extract_strings(value)
        elif isinstance(obj, list):
            for item in obj:
                extract_strings(item)
    
    extract_strings(data)
    return ' '.join(text_parts)

def detect_xai_method(explanation_text):
    text_lower = explanation_text.lower()
    
    if "sulphates" in text_lower or "sulphate" in text_lower:
        return "EBM", Path("Wine") / "wine.EBM.out.json"
    elif "middle-center" in text_lower or "middle center" in text_lower:
        return "Grad-Cam++", Path("CIFAR10") / "gradcampp_sample2_output.json"
    elif "wkhp" in text_lower:
        return "SHAP", Path("ACS_Income") / "shap_acs_output.json"
    elif "films" in text_lower or "film" in text_lower:
        return "Integrated Gradients", Path("IMDB") / "int_gradients_sample198_output.json"
    elif "carat" in text_lower:
        return "LIME", Path("Diamonds") / "LIME_diamonds.out.json"
    else:
        return None, None

def clean_markdown(text):
    text = re.sub(r'(\*\*|__|#+|\*|-|\+|\d+\.|\>)\s*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_readability(text):
    if not text or len(text.strip()) == 0:
        return None
    
    text = clean_markdown(text)
    
    try:
        metrics = {
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text)
        }
        return metrics
    except:
        return None

def process_xai_baseline(base_dir):
    base_path = Path(base_dir)
    results = {}
    skip_dirs = {'Adult (OLD)'}
    
    for dataset_dir in base_path.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name in skip_dirs:
            continue
        
        dataset_name = dataset_dir.name
        
        for json_file in dataset_dir.glob('*.json'):
            file_name = json_file.name
            text = extract_text_from_json(json_file)
            metrics = analyze_readability(text)
            
            if metrics:
                rel_path = Path(dataset_name) / file_name
                results[str(rel_path)] = metrics
    
    return results

def clean_llm_output(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        if line.strip().startswith('#') and 'EPR' in line:
            break
        cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    text = text.strip()
    
    return text

def process_llm_explanations(tp_dir, tn_dir):
    method_metrics = defaultdict(list)
    
    for directory in [tp_dir, tn_dir]:
        dir_path = Path(directory)
        
        if not dir_path.exists():
            continue        
        
        for txt_file in dir_path.glob('*.txt'):
            with open(txt_file, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            
            cleaned_text = clean_llm_output(raw_text)
            method_name, json_path = detect_xai_method(cleaned_text)
            
            if method_name is None:
                continue
            
            metrics = analyze_readability(cleaned_text)
            
            if metrics:
                method_metrics[str(json_path)].append(metrics)
    
    averaged_results = {}
    for json_path, metrics_list in method_metrics.items():
        if not metrics_list:
            continue
        
        avg_metrics = {}
        metric_keys = metrics_list[0].keys()
        
        for key in metric_keys:
            values = [m[key] for m in metrics_list]
            avg_metrics[key] = sum(values) / len(values)
        
        averaged_results[json_path] = avg_metrics
    
    return averaged_results

def get_method_name(json_path):
    if 'wine.EBM' in json_path:
        return 'EBM'
    elif 'gradcampp' in json_path:
        return 'Grad-CAM++'
    elif 'shap' in json_path:
        return 'SHAP'
    elif 'int_gradients' in json_path:
        return 'Int. Gradients'
    elif 'LIME' in json_path:
        return 'LIME'
    return 'Unknown'

def print_ascii_plots(baseline_flesch, baseline_grade, llm_results_dict, llm_pairs):
    """
    Generate ASCII-like bar plots showing improvement over baseline.
    
    Args:
        baseline_flesch: Average baseline Flesch score
        baseline_grade: Average baseline Grade score
        llm_results_dict: Dictionary of {llm_pair: {json_path: metrics}}
        llm_pairs: List of LLM pair names
    """
    
    # Calculate deltas for each LLM pair
    flesch_deltas = []
    grade_deltas = []
    
    for llm_pair in llm_pairs:
        llm_results = llm_results_dict[llm_pair]
        llm_flesch = [m['flesch_reading_ease'] for m in llm_results.values()]
        llm_grade = [m['flesch_kincaid_grade'] for m in llm_results.values()]
        
        avg_flesch = sum(llm_flesch) / len(llm_flesch)
        avg_grade = sum(llm_grade) / len(llm_grade)
        
        flesch_deltas.append((llm_pair, avg_flesch - baseline_flesch))
        grade_deltas.append((llm_pair, baseline_grade - avg_grade))  # Reversed for "lower is better"
    
    # Sort by delta (descending for both)
    flesch_deltas.sort(key=lambda x: x[1], reverse=True)
    grade_deltas.sort(key=lambda x: x[1], reverse=True)
    
    # Mapping for display names
    name_mapping = {
        'deep_gpt': 'deepseek-r1:14b',
        'gpt_qwen': 'gpt-oss:20b',
        'qwen_gpt': 'qwen3:30b'
    }
    
    # Print Flesch plot
    print("\nFlesch Reading Ease (↑ better)")
    max_flesch_delta = max(abs(d[1]) for d in flesch_deltas)
    max_bar_width = 25
    
    for llm_pair, delta in flesch_deltas:
        display_name = name_mapping.get(llm_pair, llm_pair)
        bar_length = int((abs(delta) / max_flesch_delta) * max_bar_width) if max_flesch_delta > 0 else 0
        bar = '█' * bar_length
        print(f"{delta:+6.2f}  {bar} {display_name}")
    
    # Print Grade plot
    print("\nFlesch-Kincaid Grade (↓ better)")
    max_grade_delta = max(abs(d[1]) for d in grade_deltas)
    
    for llm_pair, delta in grade_deltas:
        display_name = name_mapping.get(llm_pair, llm_pair)
        bar_length = int((abs(delta) / max_grade_delta) * max_bar_width) if max_grade_delta > 0 else 0
        bar = '█' * bar_length
        print(f"{delta:+6.2f}  {bar} {display_name}")
    
    print()

if __name__ == "__main__":
    xai_dir = "../XAI-Methods-outputs"
    
    analysis_runs = [
        ("../src/explainer/results/explanations_space_deep_gpt/processed_results/TP",
         "../src/explainer/results/explanations_space_deep_gpt/processed_results/TN",
         'deep_gpt'),
        ("../src/explainer/results/explanations_space_gpt_qwen/processed_results/TP",
         "../src/explainer/results/explanations_space_gpt_qwen/processed_results/TN",
         'gpt_qwen'),
        ("../src/explainer/results/explanations_space_qwen_gpt/processed_results/TP",
         "../src/explainer/results/explanations_space_qwen_gpt/processed_results/TN",
         'qwen_gpt'),
    ]
    
    baseline = process_xai_baseline(xai_dir)
    baseline_flesch = [m['flesch_reading_ease'] for m in baseline.values()]
    baseline_grade = [m['flesch_kincaid_grade'] for m in baseline.values()]
    
    avg_baseline_flesch = sum(baseline_flesch) / len(baseline_flesch)
    avg_baseline_grade = sum(baseline_grade) / len(baseline_grade)
    
    print("\n" + "="*80)
    print("ABSOLUTE READABILITY SCORES (Higher Flesch = Easier, Lower Grade = Easier)")
    print("="*80)
    
    print(f"\nBaseline XAI Methods:")
    print(f"  Flesch Reading Ease: {avg_baseline_flesch:.2f}")
    print(f"  Flesch-Kincaid Grade: {avg_baseline_grade:.2f}")
    
    all_llm_results = {}
    
    for tp_dir, tn_dir, llm_pair in analysis_runs:
        llm_results = process_llm_explanations(tp_dir, tn_dir)
        all_llm_results[llm_pair] = llm_results
        
        llm_flesch = [m['flesch_reading_ease'] for m in llm_results.values()]
        llm_grade = [m['flesch_kincaid_grade'] for m in llm_results.values()]
        
        print(f"\n{llm_pair}:")
        print(f"  Flesch Reading Ease: {sum(llm_flesch)/len(llm_flesch):.2f}")
        print(f"  Flesch-Kincaid Grade: {sum(llm_grade)/len(llm_grade):.2f}")
    
    print("="*80)
    
    # Generate ASCII plots
    llm_pairs = ['deep_gpt', 'gpt_qwen', 'qwen_gpt']
    print_ascii_plots(avg_baseline_flesch, avg_baseline_grade, all_llm_results, llm_pairs)
    
    methods = ['EBM', 'Grad-CAM++', 'SHAP', 'Int. Gradients', 'LIME']
    
    baseline_by_method = {}
    for json_path, metrics in baseline.items():
        method = get_method_name(json_path)
        baseline_by_method[method] = metrics