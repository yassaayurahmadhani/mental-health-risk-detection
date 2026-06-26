# 🧠 MentalTrendAI - Analisis Kesehatan Mental Media Sosial

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io)

> **Pipeline Hybrid AI untuk Deteksi Dini Risiko Kesehatan Mental dari Data Media Sosial menggunakan BERT, LSTM, dan Analisis Demografi Subjek**

---

## 📋 Daftar Isi

- [Deskripsi Proyek](#-deskripsi-proyek)
- [Arsitektur Pipeline](#-arsitektur-pipeline)
- [Struktur Repository](#-struktur-repository)
- [Instalasi](#-instalasi)
- [Cara Penggunaan](#-cara-penggunaan)
- [Dokumentasi File](#-dokumentasi-file)
- [Output & Hasil](#-output--hasil)
- [Metrik Evaluasi](#-metrik-evaluasi)

---

## 📖 Deskripsi Proyek

**MentalTrendAI** adalah sistem analisis kesehatan mental berbasis AI yang menganalisis data media sosial (Twitter/X) untuk:

1. **Filtering Konten Kesehatan Mental** - Mengidentifikasi tweet yang berkaitan dengan kesehatan mental menggunakan 220+ kata kunci dan analisis kontekstual BERT (IndoBERT)
2. **Klasifikasi Demografi Subjek** - Mengklasifikasikan pengguna menjadi REMAJA, DEWASA, atau MEDIA_BOT
3. **Analisis Sentimen** - Mengklasifikasikan sentimen (Positif/Negatif/Netral) menggunakan RoBERTa
4. **Klasifikasi Time Series LSTM** - Mendeteksi pola temporal risiko kesehatan mental

### Kategori Kesehatan Mental yang Didukung

| Kategori | Contoh Kata Kunci |
|----------|-------------------|
| `depression` | depresi, sedih, murung, putus asa |
| `anxiety` | cemas, khawatir, panik, gelisah |
| `stress` | stres, burnout, kewalahan |
| `ptsd` | trauma, flashback, mimpi buruk |
| `bipolar` | mood swing, manik, euphoria |
| `schizophrenia` | halusinasi, delusi, paranoid |
| `social_support` | dukungan mental, curhat |
| `therapy` | terapi, konseling, psikolog |
| `suicide` | bunuh diri, self-harm (SENSITIF) |
| `general_mental_health` | kesehatan mental |

---

## 🏗 Arsitektur Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                   MentalTrendAI Pipeline V6                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [1. PREPROCESSING] preprocesing_mental_health_and_subjek_demografi.py
│      ├─ Load CSV data (INA_TweetsPPKM_Raw.csv)                  │
│      ├─ Filter konten kesehatan mental (220+ keywords + BERT)  │
│      ├─ Analisis demografi subjek (REMAJA/DEWASA/MEDIA_BOT)    │
│      └─ Output: pipeline_ready.csv                              │
│                                                                  │
│  [2. SENTIMENT ANALYSIS] time_series_sentiment_final_fix.py     │
│      ├─ Load pretrained model (RoBERTa/IndoBERT)               │
│      ├─ Klasifikasi: Positif/Negatif/Netral                    │
│      └─ Evaluasi: Distribution, Confidence                     │
│                                                                  │
│  [3. TIME SERIES LSTM CLASSIFICATION]                           │
│      ├─ Feature Engineering (13+ features)                      │
│      ├─ Bidirectional LSTM architecture                         │
│      ├─ Huber loss (robust to outliers)                         │
│      ├─ Early stopping + ReduceLROnPlateau                      │
│      └─ Output: Confusion Matrix, Classification Report         │
│                                                                  │
│  [4. VISUALIZATION] streamlit_app.py                            │
│      ├─ Dashboard interaktif dengan filter                      │
│      ├─ Sentiment Distribution (Pie/Donut)                      │
│      ├─ Time Series Chart                                       │
│      └─ Subject Demographics Breakdown                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Struktur Repository

```
time-series-sentiment-classification/
│
├── 📄 preprocesing_mental_health_and_subjek_demografi.py  # Script preprocessing
├── 📄 time_series_sentiment_final_fix.py                   # Pipeline utama LSTM
├── 📄 streamlit_app.py                                     # Dashboard Streamlit
├── 📄 README.md                                            # Dokumentasi ini
│
├── 📂 data/                                                # Data hasil preprocessing
│   ├── INA_TweetsPPKM_MentalHealth_Filtered.csv           # Data terfilter
│   ├── INA_TweetsPPKM_MentalHealth_Filtered_pipeline_ready.csv
│   ├── mental_health_filter_report.json                   # Laporan filtering
│   ├── preprocessing_report.json                          # Laporan preprocessing
│   └── sample_filtered_tweets.txt                         # Sample untuk verifikasi
│
└── 📂 MentalTrendAI_Output/                               # Output pipeline
    ├── 📂 models/                                         # Model terlatih
    │   ├── lstm_model.keras                               # Model LSTM utama
    │   ├── lstm_best.keras                                # Best checkpoint
    │   ├── lstm_scaler.pkl                                # RobustScaler
    │   ├── category_encoder.pkl                           # LabelEncoder kategori
    │   ├── category_info.json                             # Info kategori
    │   ├── sentiment_config.json                          # Config sentiment
    │   └── topic_labels.json                              # Label untuk Streamlit
    │
    ├── 📂 evaluations/                                    # Hasil evaluasi
    │   ├── lstm_model_evaluation.json                     # Evaluasi LSTM
    │   ├── pipeline_summary_report.json                   # Summary pipeline
    │   └── sentiment_model_evaluation.json                # Evaluasi sentiment
    │
    └── 📂 visualizations/                                 # Visualisasi
        ├── lstm_confusion_matrix.png                      # Confusion matrix global
        ├── lstm_confusion_matrix_[category].png           # Per kategori
        └── lstm_training_loss.png                         # Plot training loss
```

---

## 🛠 Instalasi

### Requirements

```bash
pip install pandas numpy scikit-learn
pip install transformers torch sentencepiece accelerate
pip install tensorflow
pip install plotly matplotlib seaborn
pip install streamlit
pip install nltk gensim
pip install tqdm
```

### Setup NLTK

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

---

## 🚀 Cara Penggunaan

### 1. Preprocessing Data

```bash
python preprocesing_mental_health_and_subjek_demografi.py
```

**Input yang dibutuhkan:**
- `INA_TweetsPPKM_Raw.csv` - Data tweet mentah (required)
- `INA_TweetsPPKM_Labeled_Pure.csv` - Label sentiment (optional)

**Output:**
- `data/INA_TweetsPPKM_MentalHealth_Filtered_pipeline_ready.csv`

### 2. Training Pipeline Utama

```bash
python time_series_sentiment_final_fix.py
```

Atau sebagai module:

```python
from time_series_sentiment_final_fix import MentalTrendAIPipeline

pipeline = MentalTrendAIPipeline(output_dir='MentalTrendAI_Output')
report = pipeline.run_full_pipeline(
    csv_path='data/INA_TweetsPPKM_MentalHealth_Filtered_pipeline_ready.csv',
    lstm_epochs=200
)
```

### 3. Menjalankan Dashboard

```bash
streamlit run streamlit_app.py
```

---

## 📚 Dokumentasi File

### 1. `preprocesing_mental_health_and_subjek_demografi.py`

**Deskripsi:** Script preprocessing yang menggabungkan filtering kesehatan mental dan analisis demografi subjek.

#### Komponen Utama:

| Section | Fungsi |
|---------|--------|
| `load_mental_health_keywords()` | Memuat 220+ kata kunci dalam 10 kategori |
| `analyze_subject_category()` | Klasifikasi demografi (REMAJA/DEWASA/MEDIA_BOT) |
| `MentalHealthFilter` class | Filter konten + analisis kontekstual BERT |
| `PPKMPreprocessor` class | Konversi format untuk pipeline |

#### Indikator Demografi:

```python
# REMAJA: sekolah, kampus, kpop, fandom, crush, fomo, overthinking...
# DEWASA: kantor, gaji, suami, istri, pemerintah, kebijakan, bisnis...
# MEDIA_BOT: username mengandung news, media, bot, official, fess...
```

---

### 2. `time_series_sentiment_final_fix.py`

**Deskripsi:** Pipeline utama untuk analisis sentiment dan klasifikasi time series LSTM.

#### Konfigurasi LSTM:

```python
LSTM_IMPROVED_CONFIG = {
    'LSTM_UNITS': [128, 64, 32],      # Bidirectional units
    'DROPOUT_RATES': [0.15, 0.15, 0.1],
    'LEARNING_RATE': 0.001,
    'EARLY_STOPPING_PATIENCE': 40,
    'MAX_EPOCHS': 200
}
```

#### Arsitektur Model:

```
Input (seq_len, num_features)
    ↓
Bidirectional LSTM (128) + BatchNorm + Dropout(0.15)
    ↓
Bidirectional LSTM (64) + BatchNorm + Dropout(0.15)
    ↓
LSTM (32) + BatchNorm + Dropout(0.1)
    ↓
Dense(64) + Dropout(0.1) + Dense(32)
    ↓
Output (time_series_horizon)
```

#### Feature Engineering (13+ features):

1. **Original value** - Raw count
2. **Smoothed value** - Rolling average
3. **Log transformed** - Handle sparse data
4. **Lag features** - t-1, t-3, t-7
5. **Rolling statistics** - Mean, Std (window 3, 7)
6. **Cyclical encoding** - Day of week, Month

---

### 3. `streamlit_app.py`

**Deskripsi:** Dashboard web interaktif dengan Glassmorphism UI.

#### Fitur Dashboard:

- **Filter**: Rentang waktu, Kategori MH, Demografi
- **Metrics**: Total Posts, Unique Users
- **Charts**: Sentiment Distribution (Donut), Time Series, Subject Distribution

#### Klasifikasi On-the-fly:

```python
def classify_subject(text, user=''):
    # REMAJA jika teen_score > adult_score
    # DEWASA jika adult_score > teen_score
    # MEDIA_BOT jika username mengandung keywords tertentu
    # INDIVIDU_UMUM jika tidak ada indikator spesifik
```

---

## 📊 Output & Hasil

### Statistik Pipeline (Contoh)

```json
{
  "data_statistics": {
    "total_samples": 8055,
    "training_samples": 5637,
    "validation_samples": 1209,
    "test_samples": 1209
  }
}
```

### Distribusi Kategori Mental Health

| Kategori | Jumlah | Persentase |
|----------|--------|------------|
| therapy | 2,280 | 28.3% |
| suicide | 1,670 | 20.7% |
| schizophrenia | 1,515 | 18.8% |
| depression | 848 | 10.5% |
| stress | 470 | 5.8% |
| bipolar | 412 | 5.1% |
| ptsd | 359 | 4.5% |
| social_support | 304 | 3.8% |
| anxiety | 190 | 2.4% |

### Distribusi Sentiment

| Sentiment | Persentase |
|-----------|------------|
| Positif | 72.3% |
| Negatif | 19.0% |
| Netral | 8.7% |

---

## 📈 Metrik Evaluasi

### LSTM Model Evaluation

```json
{
  "diagnosis": {
    "status": "GOOD FIT",
    "is_underfitting": false,
    "is_overfitting": false
  },
  "training_info": {
    "final_training_loss": 1.3435,
    "final_validation_loss": 0.3368,
    "total_epochs": 50,
    "note": "Training loss > val loss is NORMAL for sparse time series"
  }
}
```

### Interpretasi Hasil:

- **Validation loss < 0.5**: Model performing well ✓
- **Training loss > validation loss**: Normal untuk data sparse dengan regularization
- **Early stopping**: Mencegah overfitting

---

## 🔧 Konfigurasi Lanjutan

### Threshold Diagnosis LSTM

```python
LSTM_DIAGNOSIS_THRESHOLDS = {
    'HIGH_LOSS_THRESHOLD': 3.0,
    'UNDERFITTING_RATIO': 10.0,
    'OVERFITTING_RATIO': 3.0,
    'GOOD_VAL_LOSS_THRESHOLD': 0.5,
    'GOOD_FIT_LOSS_GAP': 2.0
}
```

### Environment Variables

```bash
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=3
```

---

## 📝 Catatan Penting

1. **Sparse Time Series**: Training loss lebih tinggi dari validation loss adalah NORMAL
2. **Kategori SENSITIF**: `suicide` memerlukan penanganan khusus
3. **BERT Model**: Menggunakan `indolem/indobert-base-uncased` untuk Bahasa Indonesia
4. **Google Colab**: Script mendukung upload file via `files.upload()`

---

## 👥 Kontributor

- **MentalTrendAI Team**

## 📄 Lisensi

MIT License

---

> 🧠 *"Early detection saves lives - Deteksi dini menyelamatkan nyawa"*
