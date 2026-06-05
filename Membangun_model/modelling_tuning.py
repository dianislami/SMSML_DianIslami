"""
modelling_tuning.py
===================
Training dengan 5 Eksperimen + Hyperparameter Tuning + Manual MLflow Logging.
Level Advanced: menyimpan ke DagsHub + artefak tambahan.

Eksperimen:
    1. Random Forest (GridSearchCV)
    2. Gradient Boosting (GridSearchCV)
    3. Support Vector Machine (GridSearchCV)
    4. K-Nearest Neighbors (GridSearchCV)
    5. Voting Ensemble (RF + GB + SVM)

Cara menjalankan (local):
    python modelling_tuning.py

Cara menjalankan (DagsHub/Advanced):
    Isi DAGSHUB_REPO_OWNER dan DAGSHUB_REPO_NAME di bawah,
    lalu set env variable:
        export MLFLOW_TRACKING_USERNAME=<username>
        export MLFLOW_TRACKING_PASSWORD=<dagshub_token>
    kemudian:
        python modelling_tuning.py --dagshub

Menjalankan eksperimen tertentu saja:
    python modelling_tuning.py --experiments 1 3 5
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
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DAGSHUB_REPO_OWNER = "dianislami"
DAGSHUB_REPO_NAME  = "Dry_Bean_Classification"
EXPERIMENT_NAME    = "Dry-Bean-Tuning"
DATA_DIR           = "Dry_Bean_Dataset_preprocessing"

ALL_EXPERIMENTS = [1, 2, 3, 4, 5]

# ─────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dagshub', action='store_true',
                        help='Simpan ke DagsHub (Advanced)')
    parser.add_argument('--experiments', nargs='+', type=int,
                        default=ALL_EXPERIMENTS,
                        help='Pilih eksperimen yang dijalankan (1-5). Default: semua.')
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
# Artefak Helpers
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
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    return output_path

def save_classification_report_json(y_test, y_pred, output_path="classification_report.json"):
    report = classification_report(y_test, y_pred, output_dict=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    return output_path

def log_common_metrics(model, y_test, y_pred, X_train, y_train,
                       best_cv_score, best_params,
                       cv_folds=5, extra_params=None):
    """Log params, metrics, dan artefak ke MLflow active run."""
    # Params
    mlflow.log_params(best_params)
    mlflow.log_param("cv_folds", cv_folds)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    if extra_params:
        mlflow.log_params(extra_params)

    # Metrics
    acc       = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall    = recall_score(y_test, y_pred, average='weighted')
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')

    mlflow.log_metric("accuracy",               acc)
    mlflow.log_metric("f1_weighted",            f1)
    mlflow.log_metric("precision_weighted",     precision)
    mlflow.log_metric("recall_weighted",        recall)
    mlflow.log_metric("best_cv_accuracy",       best_cv_score)
    mlflow.log_metric("cv_mean_accuracy",       cv_scores.mean())
    mlflow.log_metric("cv_std_accuracy",        cv_scores.std())

    # Log model
    mlflow.sklearn.log_model(model, artifact_path="model")

    # Artefak: confusion matrix + classification report (semua eksperimen)
    class_labels = [str(c) for c in sorted(y_test.unique())]
    cm_path     = save_confusion_matrix(y_test, y_pred, class_labels)
    report_path = save_classification_report_json(y_test, y_pred)
    mlflow.log_artifact(cm_path,     artifact_path="plots")
    mlflow.log_artifact(report_path, artifact_path="reports")

    return acc, f1, precision, recall, cv_scores

# ─────────────────────────────────────────────
# EKSPERIMEN 1 – Random Forest (GridSearchCV)
# ─────────────────────────────────────────────
def experiment_1_random_forest(X_train, X_test, y_train, y_test):
    print("\n" + "="*60)
    print("EKSPERIMEN 1: Random Forest (GridSearchCV)")
    print("="*60)

    param_grid = {
        'n_estimators':      [100, 200],
        'max_depth':         [10, 20, None],
        'min_samples_split': [2, 5],
        'max_features':      ['sqrt', 'log2']
    }
    base = RandomForestClassifier(random_state=42, n_jobs=-1)
    gs   = GridSearchCV(base, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)

    best_model  = gs.best_estimator_
    y_pred      = best_model.predict(X_test)

    print(f"  Best params : {gs.best_params_}")
    print(f"  Best CV acc : {gs.best_score_:.4f}")

    with mlflow.start_run(run_name="Exp1-RandomForest-GridSearch"):
        acc, f1, prec, rec, cv_scores = log_common_metrics(
            best_model, y_test, y_pred, X_train, y_train,
            best_cv_score=gs.best_score_,
            best_params=gs.best_params_,
            extra_params={"model_type": "RandomForest"}
        )
        # Feature importance (khusus tree-based)
        fi_path = save_feature_importance(best_model, X_train.columns.tolist(),
                                          output_path="fi_rf.png")
        mlflow.log_artifact(fi_path, artifact_path="plots")
        run_id = mlflow.active_run().info.run_id

    _print_result("RandomForest", acc, f1, prec, rec, cv_scores, run_id)

# ─────────────────────────────────────────────
# EKSPERIMEN 2 – Gradient Boosting (GridSearchCV)
# ─────────────────────────────────────────────
def experiment_2_gradient_boosting(X_train, X_test, y_train, y_test):
    print("\n" + "="*60)
    print("EKSPERIMEN 2: Gradient Boosting (GridSearchCV)")
    print("="*60)

    param_grid = {
        'n_estimators':  [100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth':     [3, 5],
        'subsample':     [0.8, 1.0]
    }
    base = GradientBoostingClassifier(random_state=42)
    gs   = GridSearchCV(base, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_
    y_pred     = best_model.predict(X_test)

    print(f"  Best params : {gs.best_params_}")
    print(f"  Best CV acc : {gs.best_score_:.4f}")

    with mlflow.start_run(run_name="Exp2-GradientBoosting-GridSearch"):
        acc, f1, prec, rec, cv_scores = log_common_metrics(
            best_model, y_test, y_pred, X_train, y_train,
            best_cv_score=gs.best_score_,
            best_params=gs.best_params_,
            extra_params={"model_type": "GradientBoosting"}
        )
        fi_path = save_feature_importance(best_model, X_train.columns.tolist(),
                                          output_path="fi_gb.png")
        mlflow.log_artifact(fi_path, artifact_path="plots")
        run_id = mlflow.active_run().info.run_id

    _print_result("GradientBoosting", acc, f1, prec, rec, cv_scores, run_id)

# ─────────────────────────────────────────────
# EKSPERIMEN 3 – Support Vector Machine (GridSearchCV)
# ─────────────────────────────────────────────
def experiment_3_svm(X_train, X_test, y_train, y_test):
    print("\n" + "="*60)
    print("EKSPERIMEN 3: Support Vector Machine (GridSearchCV)")
    print("="*60)

    param_grid = {
        'C':      [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma':  ['scale', 'auto']
    }
    base = SVC(random_state=42, probability=True)
    gs   = GridSearchCV(base, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_
    y_pred     = best_model.predict(X_test)

    print(f"  Best params : {gs.best_params_}")
    print(f"  Best CV acc : {gs.best_score_:.4f}")

    with mlflow.start_run(run_name="Exp3-SVM-GridSearch"):
        acc, f1, prec, rec, cv_scores = log_common_metrics(
            best_model, y_test, y_pred, X_train, y_train,
            best_cv_score=gs.best_score_,
            best_params=gs.best_params_,
            extra_params={"model_type": "SVM"}
        )
        run_id = mlflow.active_run().info.run_id

    _print_result("SVM", acc, f1, prec, rec, cv_scores, run_id)

# ─────────────────────────────────────────────
# EKSPERIMEN 4 – K-Nearest Neighbors (GridSearchCV)
# ─────────────────────────────────────────────
def experiment_4_knn(X_train, X_test, y_train, y_test):
    print("\n" + "="*60)
    print("EKSPERIMEN 4: K-Nearest Neighbors (GridSearchCV)")
    print("="*60)

    param_grid = {
        'n_neighbors': [3, 5, 7, 11],
        'weights':     ['uniform', 'distance'],
        'metric':      ['euclidean', 'manhattan']
    }
    base = KNeighborsClassifier(n_jobs=-1)
    gs   = GridSearchCV(base, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_
    y_pred     = best_model.predict(X_test)

    print(f"  Best params : {gs.best_params_}")
    print(f"  Best CV acc : {gs.best_score_:.4f}")

    with mlflow.start_run(run_name="Exp4-KNN-GridSearch"):
        acc, f1, prec, rec, cv_scores = log_common_metrics(
            best_model, y_test, y_pred, X_train, y_train,
            best_cv_score=gs.best_score_,
            best_params=gs.best_params_,
            extra_params={"model_type": "KNN"}
        )
        run_id = mlflow.active_run().info.run_id

    _print_result("KNN", acc, f1, prec, rec, cv_scores, run_id)

# ─────────────────────────────────────────────
# EKSPERIMEN 5 – Voting Ensemble (RF + GB + SVM)
# ─────────────────────────────────────────────
def experiment_5_voting_ensemble(X_train, X_test, y_train, y_test):
    """
    Menggunakan parameter terbaik dari Exp 1 (RF) dan Exp 2 (GB) dan
    SVM terbaik dari Exp 3. Jika ingin meneruskan best_params dari
    eksperimen sebelumnya, simpan di dict lalu passing ke sini.
    Untuk simplisitas, digunakan parameter sensible default.
    """
    print("\n" + "="*60)
    print("EKSPERIMEN 5: Voting Ensemble (RF + GB + SVM)")
    print("="*60)

    rf  = RandomForestClassifier(n_estimators=200, max_depth=20,
                                  min_samples_split=2, max_features='sqrt',
                                  random_state=42, n_jobs=-1)
    gb  = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                      max_depth=5, subsample=1.0,
                                      random_state=42)
    svm = SVC(C=10, kernel='rbf', gamma='scale',
               probability=True, random_state=42)

    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('svm', svm)],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)

    # CV score
    cv_scores_val = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='accuracy')
    best_cv_score = cv_scores_val.mean()

    print(f"  CV mean acc : {best_cv_score:.4f}")

    best_params = {
        "rf_n_estimators": 200, "rf_max_depth": 20,
        "gb_n_estimators": 200, "gb_learning_rate": 0.1,
        "svm_C": 10, "svm_kernel": "rbf",
        "voting": "soft"
    }

    with mlflow.start_run(run_name="Exp5-VotingEnsemble-RF-GB-SVM"):
        acc, f1, prec, rec, cv_scores = log_common_metrics(
            ensemble, y_test, y_pred, X_train, y_train,
            best_cv_score=best_cv_score,
            best_params=best_params,
            extra_params={"model_type": "VotingEnsemble", "n_base_models": 3}
        )
        run_id = mlflow.active_run().info.run_id

    _print_result("VotingEnsemble (RF+GB+SVM)", acc, f1, prec, rec, cv_scores, run_id)

# ─────────────────────────────────────────────
# Helper Print
# ─────────────────────────────────────────────
def _print_result(name, acc, f1, prec, rec, cv_scores, run_id):
    print(f"\n✅ [{name}]")
    print(f"   Akurasi Test  : {acc:.4f}")
    print(f"   F1 Weighted   : {f1:.4f}")
    print(f"   Precision     : {prec:.4f}")
    print(f"   Recall        : {rec:.4f}")
    print(f"   CV Mean Acc   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"   Run ID        : {run_id}")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
EXPERIMENT_MAP = {
    1: experiment_1_random_forest,
    2: experiment_2_gradient_boosting,
    3: experiment_3_svm,
    4: experiment_4_knn,
    5: experiment_5_voting_ensemble,
}

def main():
    args = parse_args()
    setup_mlflow(args.dagshub)
    X_train, X_test, y_train, y_test = load_data()

    selected = sorted(set(args.experiments))
    print(f"\n[Main] Eksperimen yang akan dijalankan: {selected}")

    for exp_id in selected:
        if exp_id not in EXPERIMENT_MAP:
            print(f"[Warning] Eksperimen {exp_id} tidak dikenal, dilewati.")
            continue
        EXPERIMENT_MAP[exp_id](X_train, X_test, y_train, y_test)

    print("\n" + "="*60)
    print("[Done] Semua eksperimen selesai.")
    print("       Cek MLflow UI atau DagsHub untuk detail perbandingan.")
    print("="*60)


if __name__ == '__main__':
    main()