import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# use something like "Times new roman" - IEEE likes it
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Liberation Serif', 'Bitstream Vera Serif']
plt.rcParams['font.size'] = 28

feedback_data = pd.read_csv("../feedback_experiment_results_21-11-2025_12:25.csv")

palette = {
    "deepseek_gpt": "tab:blue",
    "gpt_qwen": "tab:orange",
    "qwen_gpt": "tab:green",
    "deep_qwen": "tab:red"
}

label_mapping = {
    "deepseek_gpt": "deepseek-r1:14b\n→ gpt-oss:20b",
    "gpt_qwen": "gpt-oss:20b\n→ qwen3:30b",
    "qwen_gpt": "qwen3:30b\n→ gpt-oss:20b",
    "deep_qwen": "deepseek-r1:14b\n→ qwen3:30b"
}

plt.figure(figsize=(16, 9))
sns.stripplot(
    data=feedback_data,
    x="llm_pair",
    y="iterations_to_verify",
    hue="llm_pair",
    palette=palette,
    legend=False,
    size=16,
    alpha=0.7,
    edgecolor="purple",
    linewidth=0.8,
    jitter=0.1,
    native_scale=False,
)

ax = plt.gca()
tick_locs = ax.get_xticks()
tick_labels = [label.get_text() for label in ax.get_xticklabels()]
ax.set_xticks(tick_locs)
ax.set_xticklabels([label_mapping[label] for label in tick_labels], fontsize=28)

# 10 % head-room so the mean text is never clipped
y_low, y_high = ax.get_ylim()
ax.set_ylim(y_low, y_high * 1.18)          # <-- expand the top

for i, llm_pair in enumerate(tick_labels):
    pair_data = feedback_data[feedback_data['llm_pair'] == llm_pair]
    mean_val = pair_data['iterations_to_verify'].mean()
    q75      = pair_data['iterations_to_verify'].quantile(0.75)
    max_val  = pair_data['iterations_to_verify'].max()
    med_val = pair_data['iterations_to_verify'].median()

    y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05
    ax.text(i, max_val + y_offset,
            f'μ={mean_val:.2f}\nQ₃={q75:.1f}',
            #f'μ={mean_val:.2f}',
            #f'Median={med_val:.1f}',
            ha='center', va='bottom', fontsize=28,
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='white', alpha=0.7, edgecolor='gray'))

#plt.xlabel("Explainer - Verifier Pair")
plt.xlabel("", fontsize=28)
plt.ylabel("Iterations to Verify", fontsize=28)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

# Make tick labels larger
ax.tick_params(axis='both', which='major', labelsize=28)

plt.tight_layout()
plt.savefig("refeed_mechanism_strip_plot_v3.png", dpi=400, transparent=True)
plt.savefig("refeed_mechanism_strip_plot_v3.pdf", dpi=400, transparent=True)
plt.close()