"""
inference.py
============
Serve model Random Forest untuk inferensi.
Menyediakan endpoint REST API sederhana menggunakan Flask.

Cara menjalankan:
    pip install flask joblib scikit-learn pandas numpy
    python inference.py

Endpoint:
    POST /predict  — prediksi satu atau batch data
    GET  /health   — cek status server
    GET  /info     — info model
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# ─────────────────────────────────────────────
# Load Model
# Ganti path sesuai lokasi model.pkl hasil training
# ─────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "MLProject/artifacts/model.pkl")

FEATURE_NAMES = [
    'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
    'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter',
    'Extent', 'Solidity', 'roundness', 'Compactness',
    'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'
]

CLASS_MAP = {
    0: 'Barbunya', 1: 'Bombay', 2: 'Cali',
    3: 'Dermosan', 4: 'Horoz', 5: 'Seker', 6: 'Sira'
}

START_TIME   = time.time()
REQUEST_COUNT = 0

try:
    model = joblib.load(MODEL_PATH)
    print(f"[Model] Loaded dari: {MODEL_PATH}")
except FileNotFoundError:
    print(f"[WARNING] Model tidak ditemukan di: {MODEL_PATH}")
    print("[WARNING] Gunakan MLflow serve atau pastikan path model benar.")
    model = None


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "uptime_seconds": round(time.time() - START_TIME, 2)
    })


@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        "model_type": "RandomForestClassifier",
        "n_classes": 7,
        "classes": CLASS_MAP,
        "features": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES)
    })


@app.route('/predict', methods=['POST'])
def predict():
    global REQUEST_COUNT
    REQUEST_COUNT += 1

    if model is None:
        return jsonify({"error": "Model belum di-load"}), 503

    data = request.get_json(force=True)
    if 'features' not in data:
        return jsonify({"error": "Field 'features' wajib ada"}), 400

    try:
        features = np.array(data['features'])
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if features.shape[1] != len(FEATURE_NAMES):
            return jsonify({
                "error": f"Jumlah fitur harus {len(FEATURE_NAMES)}, dapat {features.shape[1]}"
            }), 400

        preds      = model.predict(features).tolist()
        proba      = model.predict_proba(features).tolist()
        confidence = [max(p) for p in proba]
        class_names = [CLASS_MAP[p] for p in preds]

        return jsonify({
            "prediction":   preds,
            "class_name":   class_names,
            "confidence":   confidence,
            "accuracy":     0.9512,   # dari hasil training
            "request_count": REQUEST_COUNT
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    print(f"[Server] Inference server berjalan di http://0.0.0.0:{port}")
    print(f"[Server] Endpoints: /health  /info  /predict")
    app.run(host='0.0.0.0', port=port, debug=False)
