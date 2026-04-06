import numpy as np
from numpy.random import SFC64,SeedSequence, Generator
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from interpret.glassbox import ExplainableBoostingClassifier
from imblearn.over_sampling import SMOTE
from utils import create_xai_json_from_results
import os

verbose = False

def get_rng():
    entropy = 0x87351080e25cb0fad77a44a3be03b491
    base_sequence = SeedSequence(entropy)
    rng_bit_gen = SFC64(base_sequence)
    return  Generator(rng_bit_gen)

rng = get_rng()
seed = rng.integers(0,2**31-1)

wine = pd.read_csv(f"../Use-Case-datasets/Wine/WineQT.csv", index_col = "Id")
y = wine['quality']
X = wine.drop('quality', axis=1)
# state problem as binary classification
y_bin = (y >= 7).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.20, random_state=seed)

class_counts_before = y_train.value_counts(normalize=True) * 100
if verbose:
    print("Training set class distribution BEFORE SMOTE:")
    for cls, pct in class_counts_before.items():
        print(f"Class {cls}: {pct:.1f}%")

smote = SMOTE(random_state=seed)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

class_counts_after = pd.Series(y_train_res).value_counts(normalize=True) * 100
if verbose:
    print("\nTraining set class distribution AFTER SMOTE:")
    for cls, pct in class_counts_after.items():
        print(f"Class {cls}: {pct:.1f}%")

ebm = ExplainableBoostingClassifier()
ebm.fit(X_train, y_train)
auc = roc_auc_score(y_test, ebm.predict_proba(X_test)[:, 1])
print(f"AUC: {auc:.3f}")

explanations = ebm.explain_global()
explanations_data = explanations.data()
features = np.array(explanations_data['names'])
importances = np.array(explanations_data['scores'])

order = np.argsort(importances)[::-1]
features_sorted = features[order]
importances_sorted = importances[order]
#Text format
topk = min(15, len(features_sorted))
print(f"EBM Global feature importance (Mean absolute score)")
print(f"="*75)
for i in range(topk):
     print(f"{features_sorted[i]:<45} {importances_sorted[i]:>8.2f}")

out_json = create_xai_json_from_results(
    model_params=ebm.get_params(),
    num_samples=X_train.shape[0],
    global_importance=list(zip(features_sorted[:topk].tolist(),
                               importances_sorted[:topk].tolist())),
    feature_names=X_train.columns.to_list(),
    method="EBM",
    model_type="Explainable Boosting Machine",
    task="binary classification",
    target_variable="wine quality",
    dataset="wine"
)

out_fpath = os.path.join("..", "XAI-Methods-outputs", "Wine", "wine.EBM.out.json")
os.makedirs(os.path.dirname(out_fpath), exist_ok=True)
with open(out_fpath, "w") as f_json:
    f_json.write(out_json)