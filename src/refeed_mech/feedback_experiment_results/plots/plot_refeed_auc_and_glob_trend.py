import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import ast

# IEEE Style - matching other plots
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Liberation Serif', 'Bitstream Vera Serif']
plt.rcParams['font.size'] = 28

# Load refeed data
df = pd.read_csv("../feedback_experiment_results_21-11-2025_12:25.csv")
df['epr_history_explainer'] = df['epr_history_explainer'].apply(ast.literal_eval)

label_mapping = {
    "gpt_qwen": "GPT→Qwen",
    "deepseek_gpt": "DeepSeek→GPT",
    "deep_qwen": "DeepSeek→Qwen",
    "qwen_gpt": "Qwen→GPT"
}

color_mapping = {
    "gpt_qwen": "#ff7f0e",
    "deepseek_gpt": "#1f77b4",
    "deep_qwen": "#d62728",
    "qwen_gpt": "#2ca02c"
}

# ===========================================================
#  ROC–AUC PLOT
# ===========================================================

data_for_roc = []

for _, row in df.iterrows():
    first_epr = row['epr_history_explainer'][0]
    needs_correction = 1 if (row['iterations_to_verify'] > 2 or row['final_verdict'] == 'reject') else 0
    data_for_roc.append({
        'epr': first_epr,
        'needs_correction': needs_correction,
        'llm_pair': row['llm_pair']
    })

roc_df = pd.DataFrame(data_for_roc)

fig, ax = plt.subplots(figsize=(16, 9))

auc_scores = {}
for llm_pair in label_mapping.keys():
    subset = roc_df[roc_df['llm_pair'] == llm_pair]
    if len(subset) >= 10:
        y_true = subset['needs_correction'].values
        y_score = subset['epr'].values
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_scores[llm_pair] = auc(fpr, tpr)

sorted_pairs = sorted(auc_scores.items(), key=lambda x: x[1], reverse=True)

for llm_pair, auc_val in sorted_pairs:
    pretty_label = label_mapping[llm_pair]
    subset = roc_df[roc_df['llm_pair'] == llm_pair]

    y_true = subset['needs_correction'].values
    y_score = subset['epr'].values
    fpr, tpr, _ = roc_curve(y_true, y_score)

    ax.plot(fpr, tpr, lw=2.5,
            label=f'{pretty_label} ({auc_val:.2f})',
            color=color_mapping[llm_pair])

ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.4, label='Random (0.50)')

ax.set_xlabel('False Positive Rate', fontsize=28)
ax.set_ylabel('True Positive Rate', fontsize=28)
ax.legend(loc='lower right', fontsize=26, frameon=True,
          fancybox=True, shadow=True)
ax.grid(alpha=0.6, linestyle='--', linewidth=0.8)

# Add box around plot
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
# ax.spines['top'].set_linewidth(1)
# ax.spines['right'].set_linewidth(1)

ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

# textstr = 'Higher EPR → More likely\nto need correction'
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
# ax.text(0.4, 0.10, textstr, fontsize=9, verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig("epr_predictive_power.png", dpi=400, bbox_inches='tight', transparent=False)
plt.savefig("epr_predictive_power.pdf", dpi=400, bbox_inches='tight', transparent=True)
# plt.show()
plt.close()

# AUC Summary Printout
print("\n" + "="*60)
print("EPR Predictive Power: Initial Uncertainty → Refeed Need")
print("="*60)
print(f"{'LLM Pair':<20} {'AUC':<10} {'Interpretation'}")
print("-"*60)
for llm_pair, auc_val in sorted_pairs:
    interpretation = "Excellent" if auc_val > 0.65 else "Good" if auc_val > 0.55 else "Weak"
    print(f"{label_mapping[llm_pair]:<20} {auc_val:.3f}      {interpretation}")

# ===========================================================
#  PRINT SUCCESS VS FAILED COUNTS
# ===========================================================

# print("\n" + "="*60)
# print("Refeed Success Statistics per LLM Pair")
# print("="*60)
# print(f"{'LLM Pair':<20} {'Success':<10} {'Failed':<10} {'Total':<10}")
# print("-"*60)

# for llm_pair, pretty_label in label_mapping.items():
#     pair_df = df[df["llm_pair"] == llm_pair]
#     success_df = pair_df[pair_df["final_verdict"] == "accept"]
#     fail_df = pair_df[pair_df["final_verdict"] != "accept"]

#     num_success = len(success_df)
#     num_failed = len(fail_df)
#     total = len(pair_df)

#     print(f"{pretty_label:<20} {num_success:<10} {num_failed:<10} {total:<10}")

# ===========================================================
#  GLOBAL MEDIAN + IQR EPR TREND (SUCCESSFUL SAMPLES)
# ===========================================================

fig, ax = plt.subplots(figsize=(16, 9))

success_df = df[df["final_verdict"] == "accept"]
max_iters = success_df["epr_history_explainer"].apply(len).max()

padded_all = []
for h in success_df["epr_history_explainer"]:
    arr = np.array(h + [np.nan] * (max_iters - len(h)))
    padded_all.append(arr)
padded_all = np.vstack(padded_all)

median_epr = np.nanmedian(padded_all, axis=0)
q1 = np.nanpercentile(padded_all, 25, axis=0)
q3 = np.nanpercentile(padded_all, 75, axis=0)

valid_mask = ~np.isnan(median_epr)
median_epr = median_epr[valid_mask]
q1 = q1[valid_mask]
q3 = q3[valid_mask]

crop_n = min(6, len(median_epr))
median_epr = median_epr[:crop_n]
q1 = q1[:crop_n]
q3 = q3[:crop_n]
x = np.arange(1, crop_n + 1)

ax.fill_between(x, q1, q3, color="orange", alpha=0.25)
ax.plot(
    x, median_epr,
    marker='o',
    linewidth=2.5,
    markersize=10,
    color="#C07C29",
    label="Overall Median EPR"
)

ax.set_xlabel("Refeed Iteration", fontsize=28)
ax.set_ylabel("Median EPR", fontsize=28)
#ax.set_title("Global EPR Trend During Refeed", fontsize=14)

ax.set_xlim(1, 6)
ax.set_xticks(range(1, 7))

ax.grid(alpha=0.6, linestyle='--', linewidth=0.8)
ax.legend(fontsize=28, loc='upper right', frameon=True, fancybox=True, shadow=True)

# Add box around plot
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
# ax.spines['top'].set_linewidth(1)
# ax.spines['right'].set_linewidth(1)

# Make tick labels larger
ax.tick_params(axis='both', which='major', labelsize=28)

plt.tight_layout()
plt.savefig("epr_trend_global.png", dpi=400, bbox_inches='tight', transparent=False)
plt.savefig("epr_trend_global_transparent.pdf", dpi=400, bbox_inches='tight', transparent=True)
# plt.show()
plt.close()