"""Train, evaluate, and save e-commerce customer churn classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    normalize_customer_columns,
    prepare_customer_features,
)

RANDOM_STATE = 42
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)


def load_dataset(csv_path: Path, require_target: bool = True) -> tuple[pd.DataFrame, pd.Series | None]:
    data = pd.read_csv(csv_path)
    normalized = normalize_customer_columns(data)
    features = prepare_customer_features(normalized)
    if not require_target:
        return features, None

    if TARGET not in normalized.columns:
        raise ValueError(f"CSV is missing required target column: {TARGET}")
    target = pd.to_numeric(normalized[TARGET], errors="raise")
    if target.isna().any() or set(target.unique()) != {0, 1}:
        raise ValueError("Churn must contain both classes and only the values 0 and 1.")
    return features, target.astype(int)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def make_model_candidates() -> dict[str, Pipeline]:
    classifiers = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_estimators=300,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }
    return {
        name: Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "feature_selection",
                    SelectFromModel(
                        RandomForestClassifier(
                            class_weight="balanced_subsample",
                            n_estimators=100,
                            n_jobs=-1,
                            random_state=RANDOM_STATE,
                        ),
                        threshold="median",
                    ),
                ),
                ("classifier", classifier),
            ]
        )
        for name, classifier in classifiers.items()
    }


def score_predictions(
    target: pd.Series, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "f2": float(fbeta_score(target, predictions, beta=2, zero_division=0)),
        "accuracy": float(accuracy_score(target, predictions)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "pr_auc": float(average_precision_score(target, probabilities)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train churn classifiers and select the best validation F2 score."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent
        / "data_ecommerce_customer_churn_cleaned.csv",
        help="Path to the cleaned customer CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for the trained model and evaluation metrics.",
    )
    args = parser.parse_args()

    features, target = load_dataset(args.csv)
    if target is None:
        raise RuntimeError("Training data must include the Churn target.")

    x_train, x_temp, y_train, y_temp = train_test_split(
        features,
        target,
        test_size=0.40,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    x_valid, x_test, y_valid, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    validation_results: dict[str, dict[str, float]] = {}
    fitted_models: dict[str, Pipeline] = {}
    for name, model in make_model_candidates().items():
        model.fit(x_train, y_train)
        fitted_models[name] = model
        probabilities = model.predict_proba(x_valid)[:, 1]
        threshold_scores = [
            fbeta_score(
                y_valid,
                probabilities >= threshold,
                beta=2,
                zero_division=0,
            )
            for threshold in THRESHOLDS
        ]
        best_index = int(np.argmax(threshold_scores))
        best_threshold = float(THRESHOLDS[best_index])
        validation_results[name] = {
            "threshold": best_threshold,
            **score_predictions(y_valid, probabilities, best_threshold),
        }

    selected_name = max(
        validation_results,
        key=lambda name: (
            validation_results[name]["f2"],
            validation_results[name]["pr_auc"],
            validation_results[name]["recall"],
        ),
    )
    selected_threshold = validation_results[selected_name]["threshold"]
    selected_model = fitted_models[selected_name]
    test_probabilities = selected_model.predict_proba(x_test)[:, 1]
    test_metrics = score_predictions(y_test, test_probabilities, selected_threshold)

    model_path = args.output_dir / "churn_model.joblib"
    metrics_path = args.output_dir / "churn_model_metrics.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    final_model = clone(selected_model).fit(
        pd.concat([x_train, x_valid]),
        pd.concat([y_train, y_valid]),
    )
    joblib.dump(final_model, model_path)
    transformed_feature_names = final_model.named_steps["preprocessor"].get_feature_names_out()
    selected_features = transformed_feature_names[
        final_model.named_steps["feature_selection"].get_support()
    ].tolist()

    report = {
        "selected_model": selected_name,
        "selection_criteria": [
            "highest validation F2 (recall-weighted)",
            "highest validation PR-AUC as tie-breaker",
            "highest validation recall as second tie-breaker",
        ],
        "prediction_threshold": selected_threshold,
        "positive_class": "Churn = 1",
        "feature_selection": {
            "method": "SelectFromModel(RandomForestClassifier, threshold='median')",
            "selected_features": selected_features,
        },
        "dataset": str(args.csv.resolve()),
        "rows": int(len(features)),
        "split_rows": {
            "train": int(len(x_train)),
            "validation": int(len(x_valid)),
            "test": int(len(x_test)),
            "final_model_fit": int(len(x_train) + len(x_valid)),
        },
        "validation_metrics": validation_results,
        "test_metrics": test_metrics,
        "saved_model": str(model_path.resolve()),
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"Rows: {len(features)} "
        f"(train={len(x_train)}, validation={len(x_valid)}, test={len(x_test)})"
    )
    for name, metrics in validation_results.items():
        print(
            f"{name}: validation F2={metrics['f2']:.3f}, "
            f"recall={metrics['recall']:.3f}, "
            f"precision={metrics['precision']:.3f}, "
            f"threshold={metrics['threshold']:.2f}"
        )
    print(
        f"Selected model: {selected_name}; "
        f"test F2={test_metrics['f2']:.3f}, "
        f"test recall={test_metrics['recall']:.3f}, "
        f"test precision={test_metrics['precision']:.3f}"
    )
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
