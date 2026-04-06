import warnings
import yaml
from pathlib import Path
from time import perf_counter as pcnt
import numpy as np
import pandas as pd
# import shap
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import (
    train_test_split,
)
from sklearn.preprocessing import OrdinalEncoder
import lime
import lime.lime_tabular
from collections import defaultdict
from utils import parse_raw_feats, bplot, create_xai_json_from_results
import os

#warnings and output numeric format
warnings.filterwarnings("ignore")
pd.set_option("float_format", "{:.5f}".format)
verbose = True

# path and dataset file
here = Path(__file__).resolve().parent
parent = here.parent

diamonds = pd.read_csv(parent / "Use-Case-datasets/Diamonds/diamonds.csv", index_col=0)
target_variable = f"price"
X,y = diamonds.drop(target_variable, axis = 1), diamonds[[target_variable]].values.flatten()

oe = OrdinalEncoder()
cats = X.select_dtypes(exclude=np.number).columns.tolist()
X[cats] = oe.fit_transform(X[cats]).astype(np.float64)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.15, random_state=1121218
)
model = xgb.XGBRegressor(n_estimators=1000, max_depth = 6,  
                         tree_method="gpu_hist").fit(
    X_train, y_train
)

preds = model.predict(X_valid)
rmse = root_mean_squared_error(y_valid, preds)


booster_xgb = model.get_booster()
# shap_values_xgb = booster_xgb.predict(xgb.DMatrix(X_train, y_train), pred_contribs=True)
# print(f"SHAP values shape = {shap_values_xgb.shape}")



# Lime explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    mode='regression'
)

num_samples = int(0.1 * len(X_valid)) 
# num_samples = 1
feature_contributions = defaultdict(list)


def parse_lime_explanation(explanation, instance_id, y_true, y_pred):
    contributions = explanation.as_list()
    total_abs = sum(abs(c) for _, c in contributions)
    
    parsed_expl = []
    for cond, contrib in contributions:
        parsed_feat = parse_raw_feats([cond])[0]
        val = f"{parsed_feat['high_op']} {parsed_feat['high']}"
        if parsed_feat['low'] is not None: 
            val += f" and > {parsed_feat['low']}"

        parsed_expl.append({
            "feature": parsed_feat.get("feat", "-"),
            "value": val,
            "contribution": contrib,
            "direction": "positive" if contrib > 0 else "negative",
            "weight": abs(contrib) / total_abs if total_abs > 0 else 0.0
        })
    
    return {
        "instance_id": instance_id,
        "actual_value": float(y_true),
        "predicted_value": float(y_pred),
        "absolute_prediction_error": float(abs(y_true - y_pred)),
        "explanation": parsed_expl,
        "local_interpretability": {
            "confidence": getattr(explanation, "score", None),   # LIME reports local fidelity score R^2
            "num_features_used": len(contributions)
        }
    }

rng = np.random.default_rng()
chosen_local_exp = []
t0 = pcnt()
for i in range(min(num_samples, len(X_valid))):
    x_instance = X_valid[i:i+1].to_numpy()
    exp = explainer.explain_instance(
        data_row=x_instance.flatten(),
        predict_fn=model.predict,
        num_features=X_train.shape[1]
    )
    if rng.integers(0,min(num_samples, len(X_valid))) == i:
        pexp = parse_lime_explanation(exp,i,y_valid[i:i+1],model.predict(x_instance) )
        chosen_local_exp.append(pexp)
    for feat, val in exp.as_list():
        parsed_feat = parse_raw_feats([feat])[0]
        feature_contributions[parsed_feat.get(f"feat", f"_")].append(val)

print(f"chosen explanations size = {len(chosen_local_exp)}")
#  Aggregatation
lime_global_importance = {feat: np.mean(vals) for feat, vals in feature_contributions.items()}
lime_global_importance = dict(sorted(lime_global_importance.items(), key=lambda x: abs(x[1]), reverse=True))

print(f"Time for approx. LIME global feature importance: {pcnt() - t0: .2f} seconds.")
# Print / visualize ---
if verbose:
    print("Approximate LIME global feature importance:")
    for feat, val in lime_global_importance.items():
        print(f"{feat:<10} -> {val:>10.4f}")
    
# bplot(pd.DataFrame(list(lime_global_importance.items()), columns=['Feature', 'Importance']).set_index('Feature'),
#           f"lime_global_feature_importance.png")


lime_global_explanations = [(feat, abs(val)) for feat,val in lime_global_importance.items()]
out_json = create_xai_json_from_results(
    model_params=model.get_params(),
    num_samples=num_samples,
    global_importance=lime_global_explanations,
    feature_names=list(lime_global_importance.keys()),
    method="LIME",
    model_type="XGBRegressor",
    task="regression",
    target_variable="price",
    dataset="diamonds",
    local_explanations=chosen_local_exp
)


out_fpath = os.path.join("..", "XAI-Methods-outputs", "Diamonds", "LIME_diamonds.out.json")
os.makedirs(os.path.dirname(out_fpath), exist_ok=True)
with open(out_fpath, "w") as f_json:
    f_json.write(out_json)
    

# -------------------------- If want to plot, uncomment below --------------------------
# import matplotlib.pyplot as plt

# # Set font to Times New Roman
# plt.rcParams["font.family"] = "Times New Roman"

# # Data
# data = {
#     "Instance\nID": [676, 737, 169],
#     "Confidence": [0.072, 0.116, 0.116],
#     "Predicted\nValue": [3601.508, 1917.622, 1883.779],
#     "Absolute\nPrediction Error": [429.508, 63.377, 215.779],
#     "Most Prominent Features": [
#         "clarity ≤ 2.0\n5.71 < y ≤ 6.54\n3.0 < color ≤ 4.0\n5.7 < x ≤ 6.55",
#         "0.4 < carat ≤ 0.7\nclarity > 5.0\n4.72 < y ≤ 5.71\n56.0 < table ≤ 57.0",
#         "0.4 < carat ≤ 0.7\nclarity > 5.0\n4.72 < y ≤ 5.71\n3.0 < color ≤ 4.0"
#     ],
#     "Weights": [
#         "+0.42\n-0.17\n-0.15\n+0.11",
#         "-0.49\n+0.15\n-0.15\n-0.08",
#         "-0.40\n+0.18\n-0.13\n-0.11"
#     ]
# }

# df = pd.DataFrame(data)

# # Create figure with appropriate size
# fig, ax = plt.subplots(figsize=(16.5, 6))
# ax.axis('tight')
# ax.axis('off')

# # Create table
# table = ax.table(
#     cellText=df.values,
#     colLabels=df.columns,
#     cellLoc='center',
#     loc='center',
#     colWidths=[0.08, 0.12, 0.12, 0.15, 0.35, 0.12]
# )

# # Style the table
# table.auto_set_font_size(False)
# table.set_fontsize(11)
# table.scale(1, 3)

# # Style header row
# for i in range(len(df.columns)):
#     cell = table[(0, i)]
#     cell.set_facecolor('#D3D3D3')
#     cell.set_text_props(weight='bold', fontsize=24) 
#     cell.set_height(0.15)

# # Style data rows
# for i in range(1, len(df) + 1):
#     for j in range(len(df.columns)):
#         cell = table[(i, j)]
#         cell.set_facecolor('white')
#         cell.set_height(0.25)
        
#         # Left-align the "Most Prominent Features" column
#         if j == 4:
#             cell.set_text_props(ha='left', va='center', fontsize=24)
#         # Right-align the "Weights" column
#         elif j == 5:
#             cell.set_text_props(ha='right', va='center', fontsize=24)
#         else:
#             cell.set_text_props(ha='center', va='center', fontsize=24)

# # Add borders
# for key, cell in table.get_celld().items():
#     cell.set_edgecolor('black')
#     cell.set_linewidth(0.5)

# plt.tight_layout()

# # Save
# OUTPUT_PATH = "../XAI-Methods-outputs/Diamonds/"
# output_path = OUTPUT_PATH + "DIAMONDS_Lime.png"
# plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
# plt.show()