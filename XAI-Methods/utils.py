from re import (
    compile as re_compile,
    VERBOSE as re_VERBOSE
)
from matplotlib import pyplot as plt
import os
import matplotlib
import numpy as np
import json
from typing import Dict, List, Any, Tuple
from datetime import datetime


# Extract SHAP explanation in structured format for LLM consumption
def extract_shap_explanation_data(test, shap_values, feature_names, expected_value, sample_idx=0):
    sample_shap = shap_values[sample_idx]
    sample_features = test.iloc[sample_idx]
    
    feature_contributions = list(zip(feature_names, sample_shap, sample_features))
    feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    
    explanation_data = {
        "model_prediction": {
            "base_value": expected_value,
            "final_prediction": expected_value + sample_shap.sum(),
            "probability": 1 / (1 + np.exp(-(expected_value + sample_shap.sum()))),
        },
        "feature_contributions": []
    }
    
    for feature_name, shap_val, feature_val in feature_contributions[:10]:
        explanation_data["feature_contributions"].append({
            "feature": feature_name,
            "value": float(feature_val),
            "shap_contribution": float(shap_val),
            "impact": "increases" if shap_val > 0 else "decreases",
            "magnitude": abs(float(shap_val))
        })
    
    return explanation_data


def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


def parse_raw_feats(raw_feats):
    pattern = re_compile(
        r"""
        ^\s*
        (?:(?P<low>-?\d+(?:\.\d+)?)\s*<\s*)?
        (?P<feat>[a-zA-Z_][a-zA-Z0-9_]*)
        \s*
        (?P<op><=|>=|<|>)
        \s*
        (?P<high>-?\d+(?:\.\d+)?)
        (?:\s*(?P<op2><=|>=|<|>)\s*(?P<high2>-?\d+(?:\.\d+)?))?
        \s*$
        """,
        re_VERBOSE,
    )

    parsed = []
    for feat in raw_feats:
        m = pattern.match(feat)
        if not m:
            parsed.append({"raw": feat, "parsed": None})
            continue

        feat_name = m.group("feat")
        op1, high = m.group("op"), float(m.group("high"))
        low = m.group("low")

        if low and op1 and high:
            parsed.append({
                "raw": feat,
                "feat": feat_name,
                "low": float(low),
                "low_op": "<",
                "high": float(high),
                "high_op": op1
            })
        else:
            parsed.append({
                "raw": feat,
                "feat": feat_name,
                "low": None,
                "low_op": None,
                "high": high,
                "high_op": op1
            })
    return parsed


def bplot(imp,figname):
    plt.figure(figsize=(10, 5.5))
    cmap = plt.get_cmap("coolwarm")
    norm = matplotlib.colors.Normalize(vmin=imp["Importance"].min(),
                                    vmax=imp["Importance"].max())

    colors = [cmap(norm(val)) for val in imp["Importance"]]
    plt.barh(imp.index, imp["Importance"], color=colors, edgecolor="gray")

    plt.xlabel("Importance", fontsize=14)
    plt.ylabel("Features", fontsize=14)
    plt.title("Approx. Feature Importance", fontsize=16)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.gca().invert_yaxis()

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Estimated Importance", fontsize=12)

    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), "figs", figname)
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=400, bbox_inches="tight")


def create_model_info(
    model_type: str,
    task: str,
    params: Dict[str, Any] = None,
    target_variable: str = "",
    dataset: str = ""
) -> Dict[str, Any]:
    return {
        "model_info": {
            "model_type": model_type,
            "model_parameters": params,
            "task": task,
            "target_variable": target_variable,
            "dataset": dataset
        }
    }


def create_global_explanations(
    feature_importance_scores: List[Tuple[str, float]],
    method: str,
    sorted: bool = True
) -> Dict[str, Any]:
    if not sorted:
        feature_importance_scores.sort(key=lambda x: x[1], reverse=True)
    
    global_exps = []
    for rank, (feature, score) in enumerate(feature_importance_scores, 1):
        global_exps.append({
            "feature": feature,
            "importance_score": float(score),
            "rank": rank,
        })
    
    return {
        "global_explanations": {
            "method": method,
            "top_features_count": len(feature_importance_scores),
            "feature_importance": global_exps,
        }
    }


def create_metadata(
    xai_method: str,
    num_samples: int = 5000,
    feature_names: List[str] = None,
    **kwargs
) -> Dict[str, Any]:
    if feature_names is None:
        feature_names = []
    
    metadata = {
        "xai_method": xai_method,
        "num_samples": int(num_samples),
        "feature_names": feature_names,
        "timestamp": datetime.now().strftime("%a %d %b %Y, %I:%M%p")
    }
    metadata.update(kwargs)
    return {"metadata": metadata}


def create_complete_xai_json(
    model_info: Dict[str, Any],
    global_explanations: Dict[str, Any],
    metadata: Dict[str, Any],
    local_explanations: Dict[str, List[Dict[str, Any]]] = None
) -> str:
    complete_json = {
        **model_info,
        **global_explanations,
        **metadata
    }
    if local_explanations:
        complete_json.update(local_explanations)
    return json.dumps(complete_json, indent=2)


def create_xai_json_from_results(
    model_params: Dict[str, Any],
    num_samples: int,
    global_importance: List[Tuple[str, float]],
    feature_names: List[str],
    method: str,
    model_type: str,
    task: str,
    target_variable: str,
    dataset: str,
    local_explanations: List[Dict[str, Any]] = None,
    **metadata_kwargs
) -> str:
    chosen_params = {
        k: model_params[k]
        for k in model_params
        if k in ['learning_rate', 'smoothing_rounds', 'max_rounds', 'n_estimators', 'max_depth']
    }
    model_info = create_model_info(model_type, task, chosen_params, target_variable, dataset)
    global_exps = create_global_explanations(global_importance, method)
    meta = create_metadata(xai_method=method, feature_names=feature_names, num_samples=num_samples, **metadata_kwargs)
    if local_explanations:
        local_exps = {"local_explanations": local_explanations}
        return create_complete_xai_json(model_info, global_exps, meta, local_exps)
    else:
        return create_complete_xai_json(model_info, global_exps, meta)
