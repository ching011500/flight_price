"""
Utilities for building baseline and interaction feature matrices.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


MissingCategory = "Missing"


@dataclass
class InteractionConfig:
    """Definition of categorical/numerical interaction pairs."""

    cat_cat: Sequence[Tuple[str, str]] = field(default_factory=list)
    cat_num: Sequence[Tuple[str, str]] = field(default_factory=list)
    num_num: Sequence[Tuple[str, str]] = field(default_factory=list)


class FeatureBuilder:
    """
    Fits per-column encoders/scalers on the training set and generates
    baseline (1D) or interaction-augmented (2D) feature matrices.
    """

    def __init__(
        self,
        categorical_cols: Sequence[str],
        numeric_cols: Sequence[str],
        interaction_cfg: InteractionConfig | None = None,
    ) -> None:
        self.categorical_cols = list(categorical_cols)
        self.numeric_cols = list(numeric_cols)
        self.interaction_cfg = interaction_cfg or InteractionConfig()

        self.cat_categories: Dict[str, List[str]] = {}
        self.cat_dummy_columns: Dict[str, List[str]] = {}
        self.cat_pair_categories: Dict[Tuple[str, str], List[str]] = {}
        self.cat_pair_dummy_columns: Dict[Tuple[str, str], List[str]] = {}
        self.scaler = StandardScaler()
        self.num_impute_values: pd.Series | None = None
        self.num_scaled_columns: List[str] = [f"{col}_scaled" for col in self.numeric_cols]

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        """Collect category lists and fit numeric scaler on the training frame."""
        for col in self.categorical_cols:
            series = self._safe_category_series(df[col])
            categories = sorted(series.unique().tolist())
            if not categories:
                categories = [MissingCategory]
            self.cat_categories[col] = categories
            # drop-first encoding removes the first category
            self.cat_dummy_columns[col] = [f"{col}_{cat}" for cat in categories[1:]]

        numeric_frame = df[self.numeric_cols].copy()
        self.num_impute_values = numeric_frame.median(numeric_only=False)
        numeric_frame = numeric_frame.fillna(self.num_impute_values)
        self.scaler.fit(numeric_frame)

        for pair in self.interaction_cfg.cat_cat:
            a, b = pair
            combo_series = self._combo_series(df, a, b)
            categories = sorted(combo_series.unique().tolist())
            if not categories:
                categories = [MissingCategory]
            self.cat_pair_categories[pair] = categories
            self.cat_pair_dummy_columns[pair] = [
                f"{a}__{b}_{cat}" for cat in categories[1:]
            ]
        return self

    def transform(
        self,
        df: pd.DataFrame,
        add_interactions: bool = False,
    ) -> Tuple[np.ndarray, List[str]]:
        """Create baseline or interaction-augmented feature matrix."""
        cat_frames = self._categorical_frames(df)
        cat_frame = (
            pd.concat(cat_frames.values(), axis=1) if cat_frames else pd.DataFrame(index=df.index)
        )

        numeric_frame = df[self.numeric_cols].copy()
        numeric_frame = numeric_frame.fillna(self.num_impute_values)
        scaled_numeric = pd.DataFrame(
            self.scaler.transform(numeric_frame),
            columns=self.num_scaled_columns,
            index=df.index,
        )

        baseline = pd.concat([cat_frame, scaled_numeric], axis=1)
        if not add_interactions:
            return baseline.to_numpy(dtype=float), baseline.columns.tolist()

        interaction_parts: List[pd.DataFrame] = []

        # categorical x categorical interactions (explicit joint dummies)
        for pair in self.interaction_cfg.cat_cat:
            combo_series = self._combo_series(df, *pair)
            categories = self.cat_pair_categories[pair]
            combo_series = combo_series.where(combo_series.isin(categories), categories[0])
            categorical = pd.Categorical(combo_series, categories=categories)
            categorical_series = pd.Series(categorical, index=df.index)
            dummies = pd.get_dummies(
                categorical_series,
                prefix=f"{pair[0]}__{pair[1]}",
                drop_first=True,
            )
            expected_cols = self.cat_pair_dummy_columns[pair]
            dummies = dummies.reindex(columns=expected_cols, fill_value=0)
            interaction_parts.append(dummies)

        # categorical x numeric interactions (cat dummy multiplied by scaled numeric)
        for cat_col, num_col in self.interaction_cfg.cat_num:
            cat_dummy_frame = cat_frames.get(cat_col)
            if cat_dummy_frame is None:
                continue
            num_series = scaled_numeric[f"{num_col}_scaled"]
            for dummy_name in cat_dummy_frame.columns:
                interaction_parts.append(
                    pd.DataFrame(
                        {
                            f"{dummy_name}__x__{num_col}_scaled": (
                                cat_dummy_frame[dummy_name].values * num_series.values
                            )
                        },
                        index=df.index,
                    )
                )

        # numeric x numeric interactions (products of scaled numeric columns)
        for left, right in self.interaction_cfg.num_num:
            left_col = f"{left}_scaled"
            right_col = f"{right}_scaled"
            if left_col not in scaled_numeric or right_col not in scaled_numeric:
                continue
            interaction_parts.append(
                pd.DataFrame(
                    {
                        f"{left_col}__x__{right_col}": (
                            scaled_numeric[left_col].values * scaled_numeric[right_col].values
                        )
                    },
                    index=df.index,
                )
            )

        if interaction_parts:
            interaction_frame = pd.concat(interaction_parts, axis=1)
            full = pd.concat([baseline, interaction_frame], axis=1)
        else:
            full = baseline

        return full.to_numpy(dtype=float), full.columns.tolist()

    def _categorical_frames(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        cat_frames: Dict[str, pd.DataFrame] = {}
        for col in self.categorical_cols:
            if col not in self.cat_categories:
                continue
            series = self._safe_category_series(df[col])
            categories = self.cat_categories[col]
            series = series.where(series.isin(categories), categories[0])
            categorical = pd.Categorical(series, categories=categories)
            categorical_series = pd.Series(categorical, index=df.index)
            dummies = pd.get_dummies(categorical_series, prefix=col, drop_first=True)
            expected_cols = self.cat_dummy_columns[col]
            if expected_cols:
                dummies = dummies.reindex(columns=expected_cols, fill_value=0)
            cat_frames[col] = dummies
        return cat_frames

    @staticmethod
    def _safe_category_series(series: pd.Series) -> pd.Series:
        """Fill missing values and convert to string categories."""
        filled = series.fillna(MissingCategory).astype(str)
        # Guard against pandas casting "nan" strings during astype
        filled = filled.replace("nan", MissingCategory)
        return filled

    @staticmethod
    def _combo_series(df: pd.DataFrame, left: str, right: str) -> pd.Series:
        left_series = FeatureBuilder._safe_category_series(df[left])
        right_series = FeatureBuilder._safe_category_series(df[right])
        return left_series + "__" + right_series


