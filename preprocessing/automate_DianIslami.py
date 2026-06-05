"""
automate_Gln.py
===============
Script otomatisasi preprocessing Dry Bean Dataset.
Menjalankan seluruh tahapan preprocessing secara otomatis
dan menghasilkan dataset siap latih.

Cara penggunaan:
    python automate_Gln.py
    python automate_Gln.py --input ../dry-bean-dataset_raw/Dry_Bean_Dataset.xlsx --output ../dry-bean-dataset_preprocessing
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """Memuat dataset dari file Excel atau CSV."""
    print(f"[1/6] Memuat dataset dari: {filepath}")
    ext = os.path.splitext(filepath)[-1].lower()
    if ext in ['.xlsx', '.xls']:
        df = pd.read_excel(filepath)
    elif ext == '.csv':
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Format file tidak didukung: {ext}")
    print(f"      Shape awal: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 2. Hapus Missing Values
# ─────────────────────────────────────────────
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus baris yang memiliki nilai kosong."""
    before = df.shape[0]
    df = df.dropna()
    removed = before - df.shape[0]
    print(f"[2/6] Missing values: {removed} baris dihapus → Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 3. Hapus Duplikasi
# ─────────────────────────────────────────────
def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus baris duplikat."""
    before = df.shape[0]
    df = df.drop_duplicates()
    removed = before - df.shape[0]
    print(f"[3/6] Duplikasi: {removed} baris dihapus → Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 4. Tangani Outlier (IQR Clipping)
# ─────────────────────────────────────────────
def handle_outliers(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """Menangani outlier menggunakan metode IQR Clipping."""
    df = df.copy()
    total_clipped = 0
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        clipped = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        total_clipped += clipped
    print(f"[4/6] Outlier: {total_clipped} nilai di-clip (IQR method)")
    return df


# ─────────────────────────────────────────────
# 5. Label Encoding + Normalisasi + Split
# ─────────────────────────────────────────────
def encode_and_scale(df: pd.DataFrame, target_col: str, numeric_cols: list):
    """
    Melakukan label encoding pada target, standardisasi fitur,
    dan train-test split.

    Returns:
        X_train, X_test, y_train, y_test, scaler, label_encoder
    """
    # Label Encoding
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])
    print(f"[5/6] Label Encoding: {list(le.classes_)} → [0..{len(le.classes_)-1}]")

    X = df[numeric_cols].copy()

    # StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=numeric_cols)

    # Train-Test Split (80:20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Train: {X_train.shape}, Test: {X_test.shape}")

    return X_train, X_test, y_train, y_test, scaler, le


# ─────────────────────────────────────────────
# 6. Simpan Hasil
# ─────────────────────────────────────────────
def save_results(X_train, X_test, y_train, y_test, output_dir: str):
    """Menyimpan dataset hasil preprocessing ke folder output."""
    os.makedirs(output_dir, exist_ok=True)

    train_df = X_train.copy()
    train_df['Class'] = y_train

    test_df = X_test.copy()
    test_df['Class'] = y_test

    full_df = pd.concat([train_df, test_df], ignore_index=True)

    train_path = os.path.join(output_dir, 'dry_bean_train.csv')
    test_path  = os.path.join(output_dir, 'dry_bean_test.csv')
    full_path  = os.path.join(output_dir, 'dry_bean_preprocessed.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,   index=False)
    full_df.to_csv(full_path,   index=False)

    print(f"[6/6] Output disimpan ke: {output_dir}")
    print(f"      - dry_bean_train.csv       ({len(train_df)} baris)")
    print(f"      - dry_bean_test.csv        ({len(test_df)} baris)")
    print(f"      - dry_bean_preprocessed.csv ({len(full_df)} baris)")


# ─────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────
def preprocess_pipeline(input_path: str, output_dir: str):
    """Menjalankan seluruh pipeline preprocessing secara otomatis."""
    print("=" * 55)
    print("  AUTOMATED PREPROCESSING — Dry Bean Dataset")
    print("=" * 55)

    df = load_data(input_path)
    df = handle_missing(df)
    df = handle_duplicates(df)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_col   = 'Class'

    df = handle_outliers(df, numeric_cols)

    X_train, X_test, y_train, y_test, scaler, le = encode_and_scale(
        df, target_col, numeric_cols
    )

    save_results(X_train, X_test, y_train, y_test, output_dir)

    print("=" * 55)
    print("  PREPROCESSING SELESAI!")
    print("=" * 55)

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automated Preprocessing Dry Bean Dataset')
    parser.add_argument(
        '--input',
        type=str,
        default='../Dry_Bean_Dataset_raw/Dry_Bean_Dataset.xlsx',
        help='Path ke file raw dataset (.xlsx atau .csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='Dry_Bean_Dataset_preprocessing',
        help='Folder output hasil preprocessing'
    )
    args = parser.parse_args()

    preprocess_pipeline(args.input, args.output)
