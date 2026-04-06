import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# IEEE Style with larger fonts
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Liberation Serif', 'Bitstream Vera Serif']
plt.rcParams['font.size'] = 28

# extract EPR from .txt files (TNs + TPs)
def extract_epr_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    match = re.search(r'# EPR Explainer: ([\d.]+), Verifier: ([\d.]+)', content)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

# Load .txt data for a given pair
def load_pair_data(tp_dir, tn_dir):
    epr_verifier_tn, epr_verifier_tp = [], []
    epr_explainer_tn, epr_explainer_tp = [], []
    
    for txt_file in Path(tn_dir).glob("*.txt"):
        exp_epr, ver_epr = extract_epr_from_file(txt_file)
        if ver_epr is not None:
            epr_verifier_tn.append(ver_epr)
        if exp_epr is not None:
            epr_explainer_tn.append(exp_epr)
    
    for txt_file in Path(tp_dir).glob("*.txt"):
        exp_epr, ver_epr = extract_epr_from_file(txt_file)
        if ver_epr is not None:
            epr_verifier_tp.append(ver_epr)
        if exp_epr is not None:
            epr_explainer_tp.append(exp_epr)
    
    return {'verifier_tp': epr_verifier_tp, 'verifier_tn': epr_verifier_tn, 
            'explainer_tp': epr_explainer_tp, 'explainer_tn': epr_explainer_tn}

# Load DeepQwen data from CSV
def load_deepqwen_from_csv(csv_path):
    
    df = pd.read_csv(csv_path)
    tp_mask = (df['ground_truth'] == 'TP') & (df['predicted_category'] == 'TP')
    tn_mask = (df['ground_truth'] == 'TN') & (df['predicted_category'] == 'TN')
    
    return {
        'verifier_tp': df[tp_mask]['epr'].dropna().tolist(),
        'verifier_tn': df[tn_mask]['epr'].dropna().tolist()
    }

# Load data for all pairs
pairs_config = {
    'deepseek-r1:14b\n→ gpt-oss:20b': {
        'tp': '../../../explainer/results/explanations_space_deep_gpt/processed_results/TP',
        'tn': '../../../explainer/results/explanations_space_deep_gpt/processed_results/TN',
        'color': '#1f77b4',
        'type': 'txt',
        'accuracy': 90
    },
    'deepseek-r1:14b\n→ qwen3:30b': {
        'csv': '../../../explainer/results/verifier_test_results_deep_qwen/verifier_test_results.csv',
        'color': '#d62728',
        'type': 'csv',
        'accuracy': 81.8
    },
    'gpt-oss:20b\n→ qwen3:30b': {
        'tp': '../../../explainer/results/explanations_space_gpt_qwen/processed_results/TP',
        'tn': '../../../explainer/results/explanations_space_gpt_qwen/processed_results/TN',
        'color': '#ff7f0e',
        'type': 'txt',
        'accuracy': 95.21
    },
    'qwen3:30b\n→ gpt-oss:20b': {
        'tp': '../../../explainer/results/explanations_space_qwen_gpt/processed_results/TP',
        'tn': '../../../explainer/results/explanations_space_qwen_gpt/processed_results/TN',
        'color': '#2ca02c',
        'type': 'txt',
        'accuracy': 89.54
    }
}

# Load all data
data = {}
for pair_name, config in pairs_config.items():
    if config['type'] == 'txt':
        data[pair_name] = load_pair_data(config['tp'], config['tn'])
    else:
        data[pair_name] = load_deepqwen_from_csv(config['csv'])
    data[pair_name]['color'] = config['color']

# Create single plot for Verifier EPR
fig, ax = plt.subplots(figsize=(16, 9))

positions = [1, 2, 4, 5, 7, 8, 10, 11]
pair_names = list(data.keys())
colors_list = []
verifier_data = []

for pair_name in pair_names:
    pair_data = data[pair_name]
    verifier_data.extend([pair_data['verifier_tp'], pair_data['verifier_tn']])
    colors_list.extend([pair_data['color'], pair_data['color']])

# Plot Verifier EPR
parts = ax.violinplot(
    verifier_data,
    positions=positions,
    widths=0.8,
    showmeans=True,
    showmedians=True
)

for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors_list[i])
    pc.set_alpha(0.65)

ax.set_ylabel('Verifier EPR', fontsize=28)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

# Add box around plot
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)

ax.set_xticks([1.5, 4.5, 7.5, 10.5])
ax.set_xticklabels(pair_names, fontsize=28)
ax.set_xlim(0, 12)

# Add TP/TN labels with larger font
for i, pos in enumerate(positions):
    label = 'TP' if i % 2 == 0 else 'TN'
    ax.text(pos, ax.get_ylim()[0] - 0.2 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 
            label, ha='center', fontsize=28, color='#555', fontweight='bold')

# Mean EPR annotations for each pair on top of violins
body_coords = [pc.get_paths()[0].vertices for pc in parts['bodies']]
max_heights = [coords[:, 1].max() for coords in body_coords]

for i, pair_name in enumerate(pair_names):
    tp_idx = 2 * i
    tn_idx = 2 * i + 1
    
    ymax = max(max_heights[tp_idx], max_heights[tn_idx])
    x_center = (positions[tp_idx] + positions[tn_idx]) / 2
    
    # Calculate mean EPR for this pair
    pair_data = data[pair_name]
    all_epr = pair_data['verifier_tp'] + pair_data['verifier_tn']
    mean_epr = np.mean(all_epr)
    
    ax.text(
        x_center,
        ymax + 0.035 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
        f"μ: {mean_epr:.3f}",
        ha='center',
        va='bottom',
        fontsize=28,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray')
    )

# Extend y-axis upper limit to make room for annotations
ylim = ax.get_ylim()
ax.set_ylim(ylim[0], ylim[1] * 1.08)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

plt.tight_layout()
plt.savefig("verifier_epr_only.png", dpi=400, bbox_inches='tight', transparent=True)
plt.savefig("verifier_epr_only.pdf", dpi=400, bbox_inches='tight', transparent=True)
# plt.show()
plt.close()

# Print summary statistics
print("\n" + "="*70)
print("VERIFIER EPR SUMMARY (Lower = More Confident/Better)")
print("="*70)
for pair_name, pair_data in data.items():
    all_epr = pair_data['verifier_tp'] + pair_data['verifier_tn']
    print(f"{pair_name:20s}: μ={np.mean(all_epr):.3f}, σ={np.std(all_epr):.3f}")