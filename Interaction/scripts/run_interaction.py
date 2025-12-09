"""
Run baseline (1D) vs. interaction (2D) modeling for four cabin segments.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

from .feature_builder import FeatureBuilder, InteractionConfig

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR.parent / "merge_and_cleaned" / "final_data"
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
(RESULT_DIR / "plots").mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TARGET_COL = "平均價格_log"

try:
    from xgboost import XGBRegressor as NativeXGBRegressor

    USING_NATIVE_XGB = True
except ModuleNotFoundError:
    USING_NATIVE_XGB = False

    class NativeXGBRegressor(HistGradientBoostingRegressor):
        """
        Lightweight fallback when the real xgboost package is unavailable
        (e.g., offline environments). Parameters that do not exist in
        HistGradientBoostingRegressor are silently ignored.
        """

        def __init__(self, **kwargs):
            super().__init__(
                learning_rate=kwargs.get("learning_rate", 0.05),
                max_depth=kwargs.get("max_depth", 6),
                max_bins=255,
                l2_regularization=kwargs.get("reg_lambda", 1.0),
                random_state=kwargs.get("random_state", RANDOM_SEED),
            )

SHORT_CAT_COLS = [
    "出發時段",
    "星期",
    "出發機場代號",
    "抵達時段",
    "抵達機場代號",
    "航空公司",
    "航空聯盟",
    "停靠站數量",
    "是否為平日",
    "機型分類",
    "假期",
    "Region",
]
SHORT_NUM_COLS = [
    "飛行時間_分鐘",
    "價格變異",
    "最低價格剩餘天數",
    "筆數",
    "經濟指標",
    "機場指標",
    "competing_flights",
]

LONG_CAT_COLS = [
    "出發時段",
    "星期",
    "出發機場代號",
    "抵達時段",
    "抵達機場代號",
    "航空公司（主航段）",
    "航空聯盟組合",
    "航空公司組合",
    "航空聯盟",
    "停靠站數量",
    "機型分類",
    "假期",
    "飛行時間兩段分類",
    "是否為平日",
    "Region",
]
LONG_NUM_COLS = [
    "停留時間_分鐘",
    "飛行時間_分鐘",
    "實際飛行時間_分鐘",
    "價格變異",
    "最低價格剩餘天數",
    "筆數",
    "經濟指標",
    "機場指標",
    "competing_flights",
]

INTERACTION_CONFIGS = {
    "short": InteractionConfig(
        cat_cat=[
            ("出發時段", "航空聯盟"),
            ("出發時段", "假期"),
            ("出發機場代號", "航空聯盟"),
            ("抵達機場代號", "航空聯盟"),
            ("是否為平日", "航空聯盟"),
        ],
        cat_num=[
            ("出發時段", "飛行時間_分鐘"),
            ("航空聯盟", "competing_flights"),
            ("假期", "價格變異"),
        ],
        num_num=[
            ("飛行時間_分鐘", "competing_flights"),
            ("價格變異", "最低價格剩餘天數"),
            ("筆數", "competing_flights"),
        ],
    ),
    "long": InteractionConfig(
        cat_cat=[
            ("出發時段", "航空聯盟"),
            ("出發時段", "假期"),
            ("飛行時間兩段分類", "航空聯盟"),
            ("出發機場代號", "航空聯盟"),
            ("抵達機場代號", "航空聯盟"),
        ],
        cat_num=[
            ("飛行時間兩段分類", "飛行時間_分鐘"),
            ("航空聯盟", "competing_flights"),
            ("假期", "停留時間_分鐘"),
        ],
        num_num=[
            ("停留時間_分鐘", "飛行時間_分鐘"),
            ("實際飛行時間_分鐘", "competing_flights"),
            ("價格變異", "最低價格剩餘天數"),
        ],
    ),
}

CABIN_SETTINGS = [
    {
        "name": "short_eco",
        "segment": "short",
        "data_path": DATA_DIR / "short_flight.csv",
        "cabin_column": "艙等",
        "cabin_value": "經濟艙",
    },
    {
        "name": "short_biz",
        "segment": "short",
        "data_path": DATA_DIR / "short_flight.csv",
        "cabin_column": "艙等",
        "cabin_value": "商務艙",
    },
    {
        "name": "long_eco",
        "segment": "long",
        "data_path": DATA_DIR / "long_flight.csv",
        "cabin_column": "艙等（主航段）",
        "cabin_value": "經濟艙",
    },
    {
        "name": "long_biz",
        "segment": "long",
        "data_path": DATA_DIR / "long_flight.csv",
        "cabin_column": "艙等（主航段）",
        "cabin_value": "商務艙",
    },
]

MODELS = {
    "RandomForest": RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    ),
    "XGBoost": NativeXGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    ),
    "SVR": SVR(C=10.0, epsilon=0.1, gamma="scale"),
}

# Limit the number of rows used to fit SVR to avoid O(n^3) blow-up
MODEL_SAMPLE_LIMIT = {"SVR": 6000}


def run_single_segment(cfg: Dict) -> List[Dict]:
    """Train/evaluate baseline vs interaction models for a given cabin."""
    df = pd.read_csv(cfg["data_path"])
    subset = df[df[cfg["cabin_column"]] == cfg["cabin_value"]].reset_index(drop=True)
    if subset.empty:
        raise ValueError(f"No rows found for config: {cfg}")

    if cfg["segment"] == "short":
        categorical_cols = SHORT_CAT_COLS
        numeric_cols = SHORT_NUM_COLS
    else:
        categorical_cols = LONG_CAT_COLS
        numeric_cols = LONG_NUM_COLS

    missing_cols = [col for col in categorical_cols + numeric_cols if col not in subset.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns for {cfg['name']}: {missing_cols}")

    subset = subset.dropna(subset=[TARGET_COL])

    train_df, test_df = train_test_split(
        subset, test_size=0.3, random_state=RANDOM_SEED, shuffle=True
    )

    builder = FeatureBuilder(
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        interaction_cfg=INTERACTION_CONFIGS[cfg["segment"]],
    ).fit(train_df)

    X_train_base, base_cols = builder.transform(train_df, add_interactions=False)
    X_test_base, _ = builder.transform(test_df, add_interactions=False)
    X_train_inter, inter_cols = builder.transform(train_df, add_interactions=True)
    X_test_inter, _ = builder.transform(test_df, add_interactions=True)

    y_train = train_df[TARGET_COL].to_numpy()
    y_test = test_df[TARGET_COL].to_numpy()

    results: List[Dict] = []
    for feature_set, (X_train, X_test, cols) in {
        "baseline": (X_train_base, X_test_base, base_cols),
        "interaction": (X_train_inter, X_test_inter, inter_cols),
    }.items():
        for model_name, model in MODELS.items():
            estimator = clone_estimator(model)
            fit_X, fit_y, sample_info = maybe_downsample(model_name, X_train, y_train)
            t0 = perf_counter()
            estimator.fit(fit_X, fit_y)
            train_time = perf_counter() - t0

            y_pred = estimator.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = float(np.sqrt(mse))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results.append(
                {
                    "segment": cfg["segment"],
                    "cabin": cfg["cabin_value"],
                    "config_name": cfg["name"],
                    "model": model_name,
                    "feature_set": feature_set,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                    "train_rows_total": len(X_train),
                    "train_rows_used": sample_info["train_rows_used"],
                    "test_rows": len(X_test),
                    "n_features": len(cols),
                    "train_time_sec": train_time,
                }
            )
    return results


def clone_estimator(model):
    if isinstance(model, RandomForestRegressor):
        return RandomForestRegressor(**model.get_params())
    if isinstance(model, NativeXGBRegressor):
        return NativeXGBRegressor(**model.get_params())
    if isinstance(model, SVR):
        return SVR(**model.get_params())
    raise TypeError(f"Unsupported model type: {type(model)}")


def maybe_downsample(
    model_name: str, X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    limit = MODEL_SAMPLE_LIMIT.get(model_name)
    if limit is None or len(X_train) <= limit:
        return X_train, y_train, {"train_rows_used": len(X_train)}
    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.choice(len(X_train), size=limit, replace=False)
    return (
        X_train[indices],
        y_train[indices],
        {"train_rows_used": limit},
    )


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        df.pivot_table(
            index=["segment", "cabin", "model"],
            columns="feature_set",
            values=["rmse", "mae", "r2"],
        )
        .reset_index()
    )
    pivot.columns = ["_".join(col).strip("_") for col in pivot.columns.to_flat_index()]
    pivot["rmse_delta"] = pivot["rmse_interaction"] - pivot["rmse_baseline"]
    pivot["mae_delta"] = pivot["mae_interaction"] - pivot["mae_baseline"]
    pivot["r2_gain"] = pivot["r2_interaction"] - pivot["r2_baseline"]
    return pivot


def main() -> None:
    all_results: List[Dict] = []
    for cfg in CABIN_SETTINGS:
        print(f"Running segment: {cfg['name']} ...")
        seg_results = run_single_segment(cfg)
        all_results.extend(seg_results)

    result_df = pd.DataFrame(all_results)
    metrics_path = RESULT_DIR / "metrics_raw.csv"
    result_df.to_csv(metrics_path, index=False)

    summary_df = summarize_results(result_df)
    summary_path = RESULT_DIR / "metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    meta = {
        "rows": len(result_df),
        "segments": sorted({cfg["name"] for cfg in CABIN_SETTINGS}),
        "models": list(MODELS.keys()),
        "feature_sets": ["baseline", "interaction"],
        "using_native_xgb": USING_NATIVE_XGB,
    }
    (RESULT_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved detailed metrics to {metrics_path}")
    print(f"Saved summary metrics to {summary_path}")


if __name__ == "__main__":
    main()


