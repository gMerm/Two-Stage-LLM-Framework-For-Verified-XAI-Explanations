import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from fairlearn.datasets import fetch_acs_income
import json
import os
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Ensure output directory exists
OUTPUT_PATH = "../XAI-Methods-outputs/ACS_Income/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

rcParams["font.family"] = "Serif"
rcParams["font.size"] = 24

# Generate explanations for samples
def extract_shap_explanation(shap_values, sample_idx=0):
    # Handle the Explanation object (standard in newer SHAP)
    sample_shap = shap_values.values[sample_idx]
    base_val = shap_values.base_values[sample_idx]
    sample_features = shap_values.data[sample_idx]
    feature_names = shap_values.feature_names
    
    # Sort by absolute SHAP value
    indices = np.argsort(np.abs(sample_shap))[::-1][:10]
    
    # Logic for binary:logistic (log-odds to probability)
    final_pred_log_odds = base_val + sample_shap.sum()
    probability = 1 / (1 + np.exp(-final_pred_log_odds))
    
    explanation = {
        "model_prediction": {
            "base_value": float(base_val),
            "final_prediction_log_odds": float(final_pred_log_odds),
            "probability": float(probability),
        },
        "feature_contributions": []
    }
    
    for idx in indices:
        explanation["feature_contributions"].append({
            "feature": feature_names[idx],
            "value": float(sample_features[idx]),
            "shap_contribution": float(sample_shap[idx]),
            "impact": "increases" if sample_shap[idx] > 0 else "decreases",
            "magnitude": float(abs(sample_shap[idx]))
        })
    
    return explanation

# Load ACS Income dataset
data = fetch_acs_income(states=["CA"], as_frame=True)
X = data.data
y = data.target

# Median threshold to balance classes
threshold = data.target.median()
y = (data.target >= threshold).astype(int)
print(f"Using income threshold: ${threshold:,.0f}")

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model using the Scikit-Learn API
model = xgb.XGBClassifier(
    max_depth=5,
    learning_rate=0.2,
    n_estimators=100,
    objective='binary:logistic',
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")

# SHAP explanations 
# Using TreeExplainer directly on the XGBClassifier model
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

print(f"SHAP values shape: {shap_values.values.shape}")

# Plot SHAP summary bar plot
plt.figure(figsize=(12, 8))
shap.plots.bar(
    shap_values,
    max_display=10,
    show=False
)
ax = plt.gca()

# Stylize
for b in ax.patches:
    b.set_color("blue")

for txt in ax.texts:
    txt.set_color("blue")
    txt.set_fontsize(20)    
    
for label in ax.get_yticklabels():
    label.set_fontsize(20)
    label.set_fontfamily("Serif")

ax.set_title("")
ax.set_xlabel("Mean |SHAP value|", fontsize=24)
ax.set_xticks([])
plt.tight_layout()

plt.savefig(
    OUTPUT_PATH + "ACS_Income_SHAP.png",
    dpi=300,
    bbox_inches="tight",
    transparent=True
)
plt.savefig(
    OUTPUT_PATH + "ACS_Income_SHAP.pdf",
    dpi=300,
    bbox_inches="tight",
    transparent=True
)
plt.close()

# Save top features to JSON
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
feature_importance = pd.DataFrame({
    "feature": X_test.columns,
    "mean_abs_shap": mean_abs_shap
}).sort_values(by="mean_abs_shap", ascending=False)

top_features_dict = {
    "feature": feature_importance.head(10)["feature"].tolist(),
    "mean_abs_shap": feature_importance.head(10)["mean_abs_shap"].apply(float).tolist()
}

with open(OUTPUT_PATH + "shap_acs_output.json", "w") as f:
    json.dump(top_features_dict, f, indent=2)

# Generate explanations for all test samples (using first 500 for speed if needed)
all_explanations = [extract_shap_explanation(shap_values, i) for i in range(len(X_test))]
print(f"Generated explanations for {len(all_explanations)} samples")