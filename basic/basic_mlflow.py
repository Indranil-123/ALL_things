"""
============================================================
FILE 1: train_and_log.py
- Train models
- Log params & metrics to MLflow
- Save models as artifacts
- Run this FIRST
============================================================
"""

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
EXPERIMENT_NAME = "iris_classification_base"
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)

# Load data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────
def log_metrics(y_true, y_pred):
    mlflow.log_metric("accuracy",  accuracy_score(y_true, y_pred))
    mlflow.log_metric("precision", precision_score(y_true, y_pred, average="weighted"))
    mlflow.log_metric("recall",    recall_score(y_true, y_pred, average="weighted"))
    mlflow.log_metric("f1_score",  f1_score(y_true, y_pred, average="weighted"))


# ─────────────────────────────────────────────
# RUN 1 — Random Forest
# ─────────────────────────────────────────────
print("\n[RUN 1] Random Forest ...")
rf_params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}

with mlflow.start_run(run_name="RandomForest_baseline") as run1:

    mlflow.log_params(rf_params)
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("dataset",    "iris")
    mlflow.log_param("test_size",  0.2)

    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    log_metrics(y_test, y_pred)

    mlflow.set_tag("author",    "demo_user")
    mlflow.set_tag("framework", "sklearn")

    mlflow.sklearn.log_model(
        sk_model=rf,
        artifact_path="random_forest_model",
        registered_model_name="IrisRandomForest",
        input_example=X_test[:3],
    )

    with open("feature_names.txt", "w") as f:
        f.write("\n".join(iris.feature_names))
    mlflow.log_artifact("feature_names.txt")

    run1_id = run1.info.run_id
    print(f"   Run ID  : {run1_id}")
    print(f"   Accuracy: {accuracy_score(y_test, y_pred):.4f}")


# ─────────────────────────────────────────────
# RUN 2 — Logistic Regression
# ─────────────────────────────────────────────
print("\n[RUN 2] Logistic Regression ...")
lr_params = {"C": 1.0, "max_iter": 200, "solver": "lbfgs", "random_state": 42}

with mlflow.start_run(run_name="LogisticRegression_baseline") as run2:

    mlflow.log_params(lr_params)
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("dataset",    "iris")
    mlflow.log_param("scaled",     True)

    lr = LogisticRegression(**lr_params)
    lr.fit(X_train_scaled, y_train)

    y_pred = lr.predict(X_test_scaled)
    log_metrics(y_test, y_pred)

    mlflow.set_tag("author",    "demo_user")
    mlflow.set_tag("framework", "sklearn")

    mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="logistic_regression_model",
        registered_model_name="IrisLogisticRegression",
        input_example=X_test_scaled[:3],
    )

    run2_id = run2.info.run_id
    print(f"   Run ID  : {run2_id}")
    print(f"   Accuracy: {accuracy_score(y_test, y_pred):.4f}")


# ─────────────────────────────────────────────
# RUN 3 — Random Forest (tuned)
# ─────────────────────────────────────────────
print("\n[RUN 3] Random Forest (tuned) ...")
rf_params_v2 = {"n_estimators": 200, "max_depth": 10, "min_samples_split": 3, "random_state": 42}

with mlflow.start_run(run_name="RandomForest_tuned") as run3:

    mlflow.log_params(rf_params_v2)
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("version",    "v2_tuned")

    rf2 = RandomForestClassifier(**rf_params_v2)
    rf2.fit(X_train, y_train)

    for depth in [2, 5, 8, 10]:
        temp_rf = RandomForestClassifier(n_estimators=50, max_depth=depth, random_state=42)
        temp_rf.fit(X_train, y_train)
        acc = accuracy_score(y_test, temp_rf.predict(X_test))
        mlflow.log_metric("accuracy_by_depth", acc, step=depth)

    y_pred = rf2.predict(X_test)
    log_metrics(y_test, y_pred)

    mlflow.sklearn.log_model(
        sk_model=rf2,
        artifact_path="random_forest_tuned_model",
        registered_model_name="IrisRandomForest",
    )

    run3_id = run3.info.run_id
    print(f"   Run ID  : {run3_id}")
    print(f"   Accuracy: {accuracy_score(y_test, y_pred):.4f}")


# ─────────────────────────────────────────────
# SAVE RUN IDs to file so predict.py can load them
# ─────────────────────────────────────────────
run_ids = {
    "RandomForest_baseline":       run1_id,
    "LogisticRegression_baseline": run2_id,
    "RandomForest_tuned":          run3_id,
}
with open("run_ids.json", "w") as f:
    json.dump(run_ids, f, indent=2)

print("\n" + "="*55)
print("  ALL RUNS COMPLETE")
print("="*55)
print(f"  Run IDs saved to → run_ids.json")
print(f"  Now run:  python predict.py")
print(f"  MLflow UI: mlflow ui → http://localhost:5000")
print("="*55)