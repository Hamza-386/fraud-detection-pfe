"""Loading of the production models outside the API (offline scripts)."""
from __future__ import annotations

import json
import os
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


class StackingScorer:
    """CatBoost + LightGBM + XGBoost combined by the meta-model, exactly
    as api.main._predict_proba does in stacking mode."""

    def __init__(self, models_dir: str):
        with open(os.path.join(models_dir, "feature_names.json"), encoding="utf-8") as f:
            self.feature_names = json.load(f)
        with open(os.path.join(models_dir, "stacking_config.json"), encoding="utf-8") as f:
            config = json.load(f)
        self.catboost = CatBoostClassifier()
        self.catboost.load_model(os.path.join(models_dir, config["base_models"]["catboost"]))
        self.base = {name: joblib.load(os.path.join(models_dir, fn))
                     for name, fn in config["base_models"].items() if name != "catboost"}
        self.meta = joblib.load(os.path.join(models_dir, config["meta_model"]))
        self.order = list(config["meta_input_order"])

    def predict_all(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        out = {"catboost": self.catboost.predict_proba(X)[:, 1]}
        for name, m in self.base.items():
            out[name] = m.predict_proba(X)[:, 1]
        meta_in = np.column_stack([out[n] for n in self.order]).astype(np.float64)
        out["stacking"] = self.meta.predict_proba(meta_in)[:, 1]
        return out
