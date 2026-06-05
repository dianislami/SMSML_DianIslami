"""
modelling.py
============
Training model Random Forest untuk Dry Bean Classification.
Menggunakan MLflow autolog untuk tracking eksperimen.

Cara menjalankan:
    mlflow ui                    # jalankan di terminal lain
    python modelling.py
"""

import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Konfigurasi MLflow
# ─────────────────────────────────────────────
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME     = "Dry-Bean-Classification"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# ─────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────
DATA_DIR = "Dry_Bean_Dataset_preprocessing"
train_df = pd.read_csv(os.path.join(DATA_DIR, "dry_bean_train.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "dry_bean_test.csv"))

X_train = train_df.drop(columns=["Class"])
y_train = train_df["Class"]
X_test  = test_df.drop(columns=["Class"])
y_test  = test_df["Class"]

print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# ─────────────────────────────────────────────
# Training dengan MLflow Autolog
# ─────────────────────────────────────────────
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="RandomForest-autolog"):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\nAkurasi Test: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

print("\nRun selesai! Buka http://127.0.0.1:5000 untuk melihat hasil.")
