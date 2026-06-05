# SMSML_Gln — Panduan Lengkap

## Dataset
**Dry Bean Dataset**  
🔗 Download: https://www.kaggle.com/datasets/muratkokludataset/dry-bean-dataset  
File yang dibutuhkan: `Dry_Bean_Dataset.xlsx`

---

## Struktur Folder Submission

```
SMSML_Gln/
├── Eksperimen_SML_Gln.txt          ← isi link GitHub repo K1
├── Eksperimen_SML_Gln/             ← repo GitHub K1 (public)
│   ├── .github/workflows/
│   │   └── preprocessing.yml
│   ├── dry-bean-dataset_raw/
│   │   └── Dry_Bean_Dataset.xlsx   ← taruh dataset di sini
│   └── preprocessing/
│       ├── Eksperimen_Gln.ipynb
│       ├── automate_Gln.py
│       └── dry-bean-dataset_preprocessing/  ← output otomatis
│           ├── dry_bean_preprocessed.csv
│           ├── dry_bean_train.csv
│           └── dry_bean_test.csv
│
├── Membangun_model/
│   ├── modelling.py
│   ├── modelling_tuning.py
│   ├── dry-bean-dataset_preprocessing/  ← copy dari K1
│   ├── requirements.txt
│   ├── screenshoot_dashboard.jpg        ← screenshot MLflow UI
│   ├── screenshoot_artifak.jpg          ← screenshot artifacts
│   └── DagsHub.txt                      ← link DagsHub
│
├── Workflow-CI.txt                  ← isi link GitHub repo K3
├── Workflow-CI/                     ← repo GitHub K3 (public)
│   ├── .github/workflows/
│   │   └── ci.yml
│   └── MLProject/
│       ├── MLProject
│       ├── conda.yaml
│       ├── modelling.py
│       └── dry-bean-dataset_preprocessing/
│
└── Monitoring dan Logging/
    ├── 1.bukti_serving/             ← screenshot mlflow serve
    ├── 2.prometheus.yml
    ├── 3.prometheus_exporter.py
    ├── 4.bukti monitoring Prometheus/
    ├── 5.bukti monitoring Grafana/
    ├── 6.bukti alerting Grafana/
    └── 7.inference.py
```

---

## CARA PENGERJAAN STEP BY STEP

---

### ✅ KRITERIA 1 — Eksperimen Dataset

**Langkah:**

1. **Download dataset** dari Kaggle:
   - https://www.kaggle.com/datasets/muratkokludataset/dry-bean-dataset
   - Simpan `Dry_Bean_Dataset.xlsx` ke folder `Eksperimen_SML_Gln/dry-bean-dataset_raw/`

2. **Buat GitHub repo baru** bernama `Eksperimen_SML_Gln` (visibility: Public)

3. **Upload semua file** di folder `Eksperimen_SML_Gln/` ke repo

4. **Jalankan notebook** `Eksperimen_Gln.ipynb` dari awal sampai akhir tanpa error
   - Bisa di Jupyter Lab / VS Code / Google Colab

5. **Test automate.py:**
   ```bash
   cd preprocessing
   python automate_Gln.py \
     --input ../dry-bean-dataset_raw/Dry_Bean_Dataset.xlsx \
     --output dry-bean-dataset_preprocessing
   ```

6. **GitHub Actions (Advance):**
   - Push ke `main` → Actions otomatis jalan
   - Cek tab Actions di GitHub, pastikan hijau ✅
   - File preprocessing tersimpan otomatis ke repo

7. Salin link repo ke `Eksperimen_SML_Gln.txt`

---

### ✅ KRITERIA 2 — Membangun Model

**Langkah:**

1. **Install dependencies:**
   ```bash
   cd Membangun_model
   pip install -r requirements.txt
   ```

2. **Copy folder preprocessing:**
   ```bash
   cp -r ../Eksperimen_SML_Gln/preprocessing/dry-bean-dataset_preprocessing ./
   ```

3. **Jalankan MLflow UI** (terminal 1):
   ```bash
   mlflow ui
   # Buka http://127.0.0.1:5000
   ```

4. **Jalankan modelling.py** (terminal 2):
   ```bash
   python modelling.py
   ```

5. **Jalankan modelling_tuning.py** — untuk Skilled:
   ```bash
   python modelling_tuning.py
   ```
   Untuk Advanced (DagsHub):
   ```bash
   # Setup dulu di DagsHub.com → buat repo → copy token
   python modelling_tuning.py --dagshub
   ```

6. **Screenshot** MLflow dashboard dan artifacts, simpan sebagai:
   - `screenshoot_dashboard.jpg`
   - `screenshoot_artifak.jpg`

7. Untuk Advanced, buat `DagsHub.txt` berisi link repo DagsHub kamu

---

### ✅ KRITERIA 3 — Workflow CI

**Langkah:**

1. **Buat GitHub repo baru** bernama `Workflow-CI` (visibility: Public)

2. **Upload semua file** di folder `Workflow-CI/` ke repo

3. **Copy dataset preprocessing** ke `MLProject/dry-bean-dataset_preprocessing/`

4. **Setup GitHub Secrets** (di Settings → Secrets → Actions):
   ```
   MLFLOW_TRACKING_USERNAME  → username DagsHub kamu
   MLFLOW_TRACKING_PASSWORD  → token DagsHub kamu
   DAGSHUB_REPO_OWNER        → username DagsHub kamu
   DAGSHUB_REPO_NAME         → nama repo DagsHub kamu
   DOCKERHUB_USERNAME        → username Docker Hub kamu
   DOCKERHUB_TOKEN           → access token Docker Hub kamu
   ```

5. **Buat akun Docker Hub** (gratis): https://hub.docker.com
   - Settings → Security → New Access Token → copy token ke secret

6. **Trigger workflow:**
   - Push file apapun di folder `MLProject/` → CI otomatis jalan
   - Atau klik "Run workflow" di tab Actions

7. Pastikan semua step hijau ✅, Docker image muncul di Docker Hub

8. Salin link repo ke `Workflow-CI.txt`

---

### ✅ KRITERIA 4 — Monitoring & Logging

**Langkah:**

**A. Serve Model**
```bash
# Opsi 1: pakai MLflow serve
mlflow models serve \
  -m "mlruns/0/<RUN_ID>/artifacts/model" \
  -p 5001 --no-conda

# Opsi 2: pakai inference.py
pip install flask
python 7.inference.py
```

**B. Install & Jalankan Prometheus**
```bash
# Download Prometheus dari: https://prometheus.io/download/
# Extract, lalu:
./prometheus --config.file=2.prometheus.yml
# Buka: http://localhost:9090
```

**C. Jalankan Exporter**
```bash
pip install prometheus-client
python 3.prometheus_exporter.py
# Buka: http://localhost:8000/metrics
```

**D. Install & Jalankan Grafana**
```bash
# Download dari: https://grafana.com/grafana/download
# Atau via winget (Windows): winget install GrafanLabs.Grafana
# Start service, buka: http://localhost:3000
# Login: admin/admin
```

**E. Setup Grafana:**
1. Add data source → Prometheus → URL: `http://localhost:9090`
2. Buat dashboard baru, **nama dashboard = username Dicoding kamu**
3. Tambahkan panel untuk 10+ metriks:
   - `ml_request_total`
   - `ml_request_latency_seconds`
   - `ml_model_accuracy`
   - `ml_prediction_class_total`
   - `ml_prediction_confidence_avg`
   - `ml_model_memory_mb`
   - `ml_throughput_rps`
   - `ml_model_uptime_seconds`
   - `ml_batch_request_total`
   - `ml_invalid_input_total`

**F. Setup Alerting (3 alert untuk Advanced):**
1. Alert 1: Accuracy < 0.9 → `ml_model_accuracy < 0.9`
2. Alert 2: Latency > 1s → `ml_request_latency_seconds > 1`
3. Alert 3: Error rate > 5% → `rate(ml_prediction_error_total[5m]) > 0.05`

**G. Screenshot semua bukti** dan simpan ke folder yang sesuai

---

## Catatan Penting

- ⚠️ Python yang digunakan: **3.12.7**
- ⚠️ MLflow yang digunakan: **2.19.0**
- ⚠️ Nama dashboard Grafana **HARUS** pakai username Dicoding
- ⚠️ Kedua repo GitHub **HARUS** visibility Public
- ⚠️ Jalankan notebook dari awal sampai akhir tanpa error sebelum submit
