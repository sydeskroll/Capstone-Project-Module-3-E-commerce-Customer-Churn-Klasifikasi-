"""Predict customer churn from a CSV using the saved training artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from train_churn_model import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict churn labels and probabilities for customer records."
    )
    parser.add_argument("--csv", type=Path, required=True, help="Input customer CSV.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parent / "churn_model.joblib",
        help="Path to the saved classification pipeline.",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path(__file__).resolve().parent / "churn_model_metrics.json",
        help="Metrics JSON containing the selected prediction threshold.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("churn_predictions.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    customers, _ = load_dataset(args.csv, require_target=False)
    with args.metrics.open(encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)
    threshold = float(metrics["prediction_threshold"])
    model = joblib.load(args.model)

    predictions = pd.read_csv(args.csv)
    probabilities = model.predict_proba(customers)[:, 1]
    predictions["ChurnProbability"] = probabilities
    predictions["PredictedChurn"] = (probabilities >= threshold).astype(int)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)

    print(
        f"Predicted {len(predictions)} customers using threshold {threshold:.2f}; "
        f"results saved to {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
