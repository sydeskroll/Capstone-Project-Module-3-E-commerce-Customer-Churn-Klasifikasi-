# Klasifikasi Churn Pelanggan E-commerce

Proyek ini membangun dan mengevaluasi model klasifikasi biner untuk memprediksi apakah pelanggan e-commerce berpotensi churn (`Churn = 1`) atau tetap aktif (`Churn = 0`). Notebook dan pipeline pelatihan difokuskan untuk mengidentifikasi pelanggan yang membutuhkan tindakan retensi seperti promo, penawaran personalisasi, atau follow-up dari tim pemasaran atau customer success.

## Gambaran umum proyek

Analisis utama terdapat pada notebook:

- `ecommerce_customer_churn_classification.ipynb`

Proyek ini juga mencakup script pelatihan dan prediksi yang dapat digunakan kembali:

- `train_churn_model.py` – melatih dan mengevaluasi kandidat model
- `predict_churn.py` – memprediksi data pelanggan baru menggunakan model yang sudah disimpan
- `churn_model.joblib` – pipeline model klasifikasi yang sudah dilatih
- `churn_model_metrics.json` – model terpilih, threshold, dan ringkasan evaluasi

## Dataset

Model menggunakan dataset churn e-commerce yang sudah dibersihkan:

- `data_ecommerce_customer_churn_cleaned.csv`

Dataset ini mencakup fitur profil dan perilaku pelanggan, seperti:

- `Tenure`
- `WarehouseToHome`
- `NumberOfDeviceRegistered`
- `SatisfactionScore`
- `NumberOfAddress`
- `Complaint`
- `DaySinceLastOrder`
- `CashbackAmount`
- `PreferedOrderCat`
- `MaritalStatus`
- `Churn` (target label)

## Pendekatan pemodelan

Workflow yang digunakan mengikuti pipeline pembelajaran mesin standar:

1. Memuat dataset pelanggan yang sudah dibersihkan.
2. Membagi data menjadi set pelatihan, validasi, dan uji.
3. Memproses fitur numerik dan kategorikal secara terpisah:
   - imputasi median untuk data numerik
   - imputasi nilai paling sering untuk data kategorikal
   - standard scaling pada fitur numerik
   - one-hot encoding pada fitur kategorikal
4. Melatih beberapa kandidat model:
   - Logistic Regression
   - Random Forest
5. Memilih model terbaik berdasarkan performa pada data validasi dengan metrik F2, dengan PR-AUC dan recall sebagai tie-breaker.
6. Menentukan threshold keputusan terbaik untuk prediksi churn.
7. Mengevaluasi model akhir pada set test.

## Model terpilih dan metrik

Model akhir yang terpilih adalah Random Forest dengan threshold prediksi sebesar `0.36`.

Performa pada set test:

- Akurasi: `0.890`
- Presisi: `0.626`
- Recall: `0.813`
- F1-score: `0.707`
- F2-score: `0.767`
- ROC-AUC: `0.935`
- PR-AUC: `0.803`

Pengaturan ini memprioritaskan recall dan F2 score, yang sesuai untuk deteksi churn karena false negative (pelanggan churn yang terlewat) biasanya lebih mahal daripada false positive.

## Struktur repository

```text
.
├── README.md
├── ecommerce_customer_churn_classification.ipynb
├── data_ecommerce_customer_churn_cleaned.csv
├── train_churn_model.py
├── predict_churn.py
├── churn_model.joblib
├── churn_model_metrics.json
├── requirements.txt
├── E-commerce Customer Churn.docx
└── churn_model.joblib
```

## Instalasi

Buat environment virtual lalu install dependency:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# atau .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Melatih model

Untuk melatih ulang model churn dari dataset yang sudah dibersihkan:

```bash
python train_churn_model.py
```

Perintah ini akan menyimpan model ke `churn_model.joblib` dan menulis hasil evaluasi ke `churn_model_metrics.json`.

## Melakukan prediksi

Untuk memprediksi data pelanggan dari file CSV:

```bash
python predict_churn.py --csv data_ecommerce_customer_churn_cleaned.csv --output churn_predictions.csv
```

Script ini akan menghasilkan file output dengan kolom:

- `ChurnProbability`
- `PredictedChurn`

## Kegunaan bisnis

Proyek ini dapat digunakan oleh perusahaan e-commerce untuk mengidentifikasi pelanggan yang berisiko churn sebelum mereka benar-benar meninggalkan bisnis. Dengan memeringkat pelanggan berdasarkan probabilitas churn, tim marketing dan customer success dapat fokus pada pelanggan yang paling berpotensi membutuhkan tindakan retensi.

## Catatan

- Notebook merupakan artefak utama untuk eksplorasi dan penjelasan hasil analisis.
- Script Python memungkinkan model dilatih ulang dan digunakan kembali dalam workflow produksi atau pelaporan.
- Dataset sudah bersih dan siap digunakan untuk pemodelan; nama fitur pada notebook juga telah dinormalisasi untuk menghindari inkonsistensi.
