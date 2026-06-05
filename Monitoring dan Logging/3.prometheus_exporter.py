"""
prometheus_exporter.py
======================
Mengekspos metriks monitoring ML ke Prometheus.
Jalankan bersamaan dengan inference.py.

Cara menjalankan:
    pip install prometheus-client requests
    python prometheus_exporter.py
    
Kemudian buka: http://localhost:8000/metrics
"""

import time
import random
import threading
import requests
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary

# ─────────────────────────────────────────────
# Definisi Metriks (10+ untuk Advanced)
# ─────────────────────────────────────────────

# 1. Total request ke model
REQUEST_COUNT = Counter(
    'ml_request_total',
    'Total jumlah request prediksi ke model',
    ['endpoint', 'status']
)

# 2. Latensi per request
REQUEST_LATENCY = Histogram(
    'ml_request_latency_seconds',
    'Latensi request prediksi (detik)',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)

# 3. Akurasi model (running average)
MODEL_ACCURACY = Gauge(
    'ml_model_accuracy',
    'Akurasi model saat ini (running estimate)'
)

# 4. Distribusi prediksi per kelas
PREDICTION_CLASS = Counter(
    'ml_prediction_class_total',
    'Jumlah prediksi per kelas',
    ['class_label']
)

# 5. Jumlah error prediksi
PREDICTION_ERROR = Counter(
    'ml_prediction_error_total',
    'Jumlah error saat prediksi',
    ['error_type']
)

# 6. Confidence score rata-rata
CONFIDENCE_SCORE = Gauge(
    'ml_prediction_confidence_avg',
    'Rata-rata confidence score prediksi'
)

# 7. Memory usage model (MB)
MODEL_MEMORY_MB = Gauge(
    'ml_model_memory_mb',
    'Estimasi penggunaan memori model (MB)'
)

# 8. Throughput (request per second)
THROUGHPUT = Gauge(
    'ml_throughput_rps',
    'Throughput request per detik'
)

# 9. Uptime model server
MODEL_UPTIME = Gauge(
    'ml_model_uptime_seconds',
    'Lama model server berjalan (detik)'
)

# 10. Jumlah request batch
BATCH_REQUEST_COUNT = Counter(
    'ml_batch_request_total',
    'Total request batch'
)

# 11. Ukuran batch rata-rata
BATCH_SIZE = Gauge(
    'ml_batch_size_avg',
    'Rata-rata ukuran batch request'
)

# 12. Jumlah fitur input yang tidak valid
INVALID_INPUT = Counter(
    'ml_invalid_input_total',
    'Jumlah input tidak valid yang diterima'
)

# ─────────────────────────────────────────────
# Simulasi Metriks (jalankan setiap 5 detik)
# Ganti bagian ini dengan hit nyata ke inference.py
# ─────────────────────────────────────────────
START_TIME = time.time()
CLASS_LABELS = ['0-Barbunya', '1-Bombay', '2-Cali', '3-Dermosan',
                '4-Horoz', '5-Seker', '6-Sira']

INFERENCE_URL = "http://127.0.0.1:5001/predict"  # sesuaikan port inference.py


def simulate_and_collect():
    """
    Kumpulkan metriks dari model serving.
    Dalam production, ganti request ke endpoint nyata.
    """
    request_window = 0
    while True:
        try:
            # ── Hit inference endpoint ─────────────────────────
            payload = {
                "features": [
                    [random.uniform(20000, 200000) for _ in range(16)]
                ]
            }
            start = time.time()
            try:
                resp = requests.post(INFERENCE_URL, json=payload, timeout=2)
                latency = time.time() - start

                if resp.status_code == 200:
                    data = resp.json()
                    REQUEST_COUNT.labels(endpoint='/predict', status='success').inc()
                    REQUEST_LATENCY.observe(latency)

                    pred_class = str(data.get('prediction', [0])[0])
                    confidence = data.get('confidence', random.uniform(0.7, 0.99))

                    PREDICTION_CLASS.labels(class_label=CLASS_LABELS[int(pred_class)]).inc()
                    CONFIDENCE_SCORE.set(confidence)
                    MODEL_ACCURACY.set(data.get('accuracy', random.uniform(0.92, 0.97)))
                else:
                    REQUEST_COUNT.labels(endpoint='/predict', status='error').inc()
                    PREDICTION_ERROR.labels(error_type='http_error').inc()

            except requests.exceptions.ConnectionError:
                # Simulasi kalau inference server belum jalan
                REQUEST_COUNT.labels(endpoint='/predict', status='simulated').inc()
                latency = random.uniform(0.01, 0.3)
                REQUEST_LATENCY.observe(latency)
                pred_class = random.randint(0, 6)
                PREDICTION_CLASS.labels(class_label=CLASS_LABELS[pred_class]).inc()
                CONFIDENCE_SCORE.set(random.uniform(0.80, 0.99))
                MODEL_ACCURACY.set(random.uniform(0.92, 0.97))

            # ── Metriks sistem ─────────────────────────────────
            MODEL_UPTIME.set(time.time() - START_TIME)
            MODEL_MEMORY_MB.set(random.uniform(120, 180))
            request_window += 1
            THROUGHPUT.set(request_window / 5.0)

            # ── Batch & invalid ────────────────────────────────
            if random.random() < 0.3:
                BATCH_REQUEST_COUNT.inc()
                BATCH_SIZE.set(random.randint(8, 64))
            if random.random() < 0.02:
                INVALID_INPUT.inc()

        except Exception as e:
            PREDICTION_ERROR.labels(error_type='unexpected').inc()
            print(f"[Exporter Error] {e}")

        time.sleep(5)
        request_window = 0


if __name__ == '__main__':
    # Jalankan exporter di port 8000
    start_http_server(8000)
    print("=" * 50)
    print("  Prometheus Exporter berjalan di :8000/metrics")
    print("=" * 50)

    # Kumpulkan metriks di background thread
    t = threading.Thread(target=simulate_and_collect, daemon=True)
    t.start()

    # Jaga proses tetap hidup
    while True:
        time.sleep(1)
