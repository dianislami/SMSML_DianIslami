"""
modelling_tuning.py
===================
Training dengan Hyperparameter Tuning (GridSearchCV) + Manual MLflow Logging.
Level Advanced: menyimpan ke DagsHub + artefak tambahan.

Cara menjalankan (local):
    python modelling_tuning.py

Cara menjalankan (DagsHub/Advanced):
    Isi DAGSHUB_REPO_OWNER dan DAGSHUB_REPO_NAME di bawah,
    lalu set env variable:
        export MLFLOW_TRACKING_USERNAME=<username>
        export MLFLOW_TRACKING_PASSWORD=<dagshub_token>
    kemudian:
        python modelling_tuning.py --dagshub
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DAGSHUB_REPO_OWNER = "username_dagshub_kamu"   # ganti ini
DAGSHUB_REPO_NAME  = "Dry-Bean-Classification"  # ganti ini
EXPERIMENT_NAME    = "Dry-Bean-Tuning"
DATA_DIR           = "Dry_Bean_Dataset_preprocessing"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dagshub', action='store_true',
                        help='Simpan ke DagsHub (Advanced)')
    return parser.parse_args()

# ─────────────────────────────────────────────
# Setup MLflow Tracking
# ─────────────────────────────────────────────
def setup_mlflow(use_dagshub: bool):
    if use_dagshub:
        import dagshub
        dagshub.init(
            repo_owner=DAGSHUB_REPO_OWNER,
            repo_name=DAGSHUB_REPO_NAME,
            mlflow=True
        )
        print(f"[MLflow] Tracking ke DagsHub: {DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}")
    else:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        print("[MLflow] Tracking ke localhost:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)

# ─────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────
def load_data():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "dry_bean_train.csv"))
    test_df  = pd.read_csv(os.path.join(DATA_DIR, "dry_bean_test.csv"))
    X_train  = train_df.drop(columns=["Class"])
    y_train  = train_df["Class"]
    X_test   = test_df.drop(columns=["Class"])
    y_test   = test_df["Class"]
    print(f"[Data] Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

# ─────────────────────────────────────────────
# Hyperparameter Tuning
# ─────────────────────────────────────────────
def run_tuning(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth':    [10, 20, None],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt', 'log2']
    }
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        base_model, param_grid,
        cv=5, scoring='accuracy',
        n_jobs=-1, verbose=1
    )
    print("[Tuning] Menjalankan GridSearchCV...")
    grid_search.fit(X_train, y_train)
    print(f"[Tuning] Best params : {grid_search.best_params_}")
    print(f"[Tuning] Best CV acc : {grid_search.best_score_:.4f}")
    return grid_search

# ─────────────────────────────────────────────
# Artefak Tambahan (Advanced)
# ─────────────────────────────────────────────
def save_confusion_matrix(y_test, y_pred, labels, output_path="confusion_matrix.png"):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    return output_path

def save_feature_importance(model, feature_names, output_path="feature_importance.png"):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importances)), importances[indices], color='steelblue')
    plt.xticks(range(len(importances)),
               [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.title('Feature Importance (Random Forest)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    return output_path

def save_classification_report_json(y_test, y_pred, output_path="classification_report.json"):
    report = classification_report(y_test, y_pred, output_dict=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    return output_path

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    setup_mlflow(args.dagshub)

    X_train, X_test, y_train, y_test = load_data()
    grid_search = run_tuning(X_train, y_train)

    best_model  = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Prediksi
    y_pred      = best_model.predict(X_test)
    y_pred_prob = best_model.predict_proba(X_test)

    # Metriks
    acc       = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall    = recall_score(y_test, y_pred, average='weighted')
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')

    # Artefak
    class_labels  = [str(c) for c in sorted(y_test.unique())]
    cm_path       = save_confusion_matrix(y_test, y_pred, class_labels)
    fi_path       = save_feature_importance(best_model, X_train.columns.tolist())
    report_path   = save_classification_report_json(y_test, y_pred)

    # ── MLflow Manual Logging ──────────────────
    with mlflow.start_run(run_name="RF-GridSearchCV-best"):

        # Log hyperparameter
        mlflow.log_params(best_params)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        # Log metriks utama (sama dengan autolog)
        mlflow.log_metric("accuracy",       acc)
        mlflow.log_metric("f1_weighted",    f1)
        mlflow.log_metric("precision_weighted", precision)
        mlflow.log_metric("recall_weighted",    recall)
        mlflow.log_metric("best_cv_accuracy",   grid_search.best_score_)
        mlflow.log_metric("cv_mean_accuracy",   cv_scores.mean())
        mlflow.log_metric("cv_std_accuracy",    cv_scores.std())

        # Log model
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        # ── Artefak tambahan (Advanced: minimal 2) ──
        mlflow.log_artifact(cm_path,     artifact_path="plots")
        mlflow.log_artifact(fi_path,     artifact_path="plots")
        mlflow.log_artifact(report_path, artifact_path="reports")

        print(f"\n[MLflow] Run ID: {mlflow.active_run().info.run_id}")

    print(f"\n✅ Akurasi Test  : {acc:.4f}")
    print(f"   F1 Weighted   : {f1:.4f}")
    print(f"   Precision     : {precision:.4f}")
    print(f"   Recall        : {recall:.4f}")
    print(f"   CV Mean Acc   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("\n[Done] Cek MLflow UI atau DagsHub untuk detail.")


if __name__ == '__main__':
    main()
