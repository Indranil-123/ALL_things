"""
predict.py
==========
Loads the saved model from MLflow artifact store
and generates predictions on new data.

Run AFTER train_and_log.py
"""

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_iris

# ─────────────────────────────────────────────
# MUST match your tracking URI
# ─────────────────────────────────────────────
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# ─────────────────────────────────────────────
# Load using model ID — copied from artifact_path
# in MLflow UI → Artifacts → MLmodel
# ─────────────────────────────────────────────  # ← from your screenshot

model_uri = "mlflow-artifacts:/1/models/m-075422919a7943128778194caf6413c9/artifacts"
model = mlflow.sklearn.load_model(model_uri)
print(f"Loading model from: {model_uri}")


# ─────────────────────────────────────────────
# SAMPLE DATA — Iris features
# [sepal length, sepal width, petal length, petal width]
# ─────────────────────────────────────────────
iris        = load_iris()
CLASS_NAMES = iris.target_names   # ['setosa', 'versicolor', 'virginica']

# Use a few test samples from the dataset
sample_data = np.array([
    [5.1, 3.5, 1.4, 0.2],   # → setosa
    [6.7, 3.1, 4.7, 1.5],   # → versicolor
    [6.3, 3.3, 6.0, 2.5],   # → virginica
    [4.9, 3.0, 1.4, 0.2],   # → setosa
    [5.8, 2.7, 5.1, 1.9],   # → virginica
])

# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
predictions  = model.predict(sample_data)
probabilities = model.predict_proba(sample_data)

print("\n" + "="*55)
print("  PREDICTIONS")
print("="*55)
print(f"{'#':<4} {'Features':<30} {'Prediction':<14} {'Confidence'}")
print("-"*55)

for i, (row, pred, prob) in enumerate(zip(sample_data, predictions, probabilities)):
    class_name = CLASS_NAMES[pred]
    confidence = prob[pred]
    features   = str(row.tolist())
    print(f"{i+1:<4} {features:<30} {class_name:<14} {confidence:.2%}")

print("\n" + "="*55)
print("  FULL PROBABILITY BREAKDOWN")
print("="*55)
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    print(f"\nSample {i+1}  →  Predicted: {CLASS_NAMES[pred]}")
    for cls, p in zip(CLASS_NAMES, prob):
        bar = "█" * int(p * 30)
        print(f"  {cls:<12} {p:.4f}  {bar}")