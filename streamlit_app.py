# -*- coding: utf-8 -*-
"""
MentalTrendAI - Streamlit Web Application (Enhanced Dashboard)
================================================================

File: streamlit_app.py
Versi: 2.0 (Subject Filter Integration)
Terakhir diupdate: 2025-01-02

FITUR BARU:
    1. Integrasi Subject Demographics Filter (REMAJA/DEWASA/MEDIA_BOT)
    2. Mental Health Category Breakdown per Demografi
    3. Pipeline Training Results & Model Evaluation Display
    4. Enhanced Visualizations dengan Glassmorphism UI
"""

# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================
import streamlit as st
import os
import warnings
import json

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import logging
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pickle

from tensorflow.keras.models import load_model as keras_load_model

# =============================================================================
# SECTION 2: PAGE CONFIG & MODERN CSS
# =============================================================================
st.set_page_config(
    page_title="MentalTrendAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 100%);
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #FFFFFF;
    }
    
    .modern-header {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%);
        backdrop-filter: blur(20px);
        padding: 2.5rem;
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
    }
    .modern-header h1 {
        font-weight: 800;
        font-size: 3rem;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        text-shadow: 0 0 30px rgba(129, 140, 248, 0.5);
    }
    .modern-header p {
        color: #E2E8F0;
        font-size: 1.1rem;
        margin-top: 0.75rem;
        opacity: 0.9;
    }
    
    /* Glass Card */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: rgba(129, 140, 248, 0.5);
        box-shadow: 0 20px 40px rgba(129, 140, 248, 0.15);
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(51, 65, 85, 0.6) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 16px;
        transition: all 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        border-color: #818CF8;
        box-shadow: 0 10px 30px rgba(129, 140, 248, 0.2);
    }
    
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] div, [data-testid="stMetricLabel"] span {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }
    
    div[data-testid="stMetricValue"] > div {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        font-size: 2rem !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617 0%, #0F172A 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    section[data-testid="stSidebar"] * { color: #F8FAFC !important; }
    
    /* Section Headers */
    .section-header {
        color: #FFFFFF;
        font-weight: 700;
        font-size: 1.1rem;
        margin: 1.5rem 0 1rem 0;
        padding-left: 12px;
        border-left: 4px solid;
        border-image: linear-gradient(to bottom, #38BDF8, #818CF8) 1;
    }
    
    /* Stats Badge */
    .stats-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 0.25rem;
    }
    
    /* Dropdown/Selectbox Styling */
    div[data-baseweb="select"] {
        background: linear-gradient(135deg, rgba(71, 85, 105, 0.9) 0%, rgba(100, 116, 139, 0.8) 100%) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(148, 163, 184, 0.5) !important;
    }
    div[data-baseweb="select"]:hover {
        border-color: #818CF8 !important;
        box-shadow: 0 4px 15px rgba(129, 140, 248, 0.3) !important;
    }
    div[data-baseweb="select"] > div {
        background: transparent !important;
        color: #1E293B !important;
    }
    div[data-baseweb="select"] input {
        color: #1E293B !important;
    }
    div[data-baseweb="select"] span {
        color: #1E293B !important;
    }
    /* Dropdown menu styling */
    div[data-baseweb="popover"] {
        background: rgba(241, 245, 249, 0.98) !important;
        border: 1px solid rgba(148, 163, 184, 0.4) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px) !important;
    }
    div[data-baseweb="popover"] li {
        color: #1E293B !important;
    }
    div[data-baseweb="popover"] li:hover {
        background: rgba(129, 140, 248, 0.3) !important;
    }
    /* Selectbox label styling */
    .stSelectbox label {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SECTION 3: SUBJECT CLASSIFICATION (On-the-fly)
# =============================================================================

# Indicators for subject classification
TEEN_INDICATORS = [
    'sekolah', 'guru', 'tugas', 'ujian', 'ulangan', 'semester', 'krs', 'ipk', 
    'dosen', 'kampus', 'skripsi', 'mahasiswa', 'snmptn', 'sbmptn', 'utbk',
    'kpop', 'bias', 'fandom', 'drakor', 'anime', 'manga', 'webtoon', 'wattpad',
    'bestie', 'nangis', 'fomo', 'overthinking', 'insecure', 'crush', 'confess',
    'menfess', 'spill', 'cringe', 'red flag', 'healing', 'moots', 'mutual'
]

ADULT_INDICATORS = [
    'kantor', 'kerjaan', 'lembur', 'meeting', 'rapat', 'dinas', 'gaji', 
    'rekening', 'transfer', 'bank', 'cicilan', 'kredit', 'tagihan', 'pajak',
    'npwp', 'bpjs', 'hrd', 'boss', 'atasan', 'klien', 'proyek', 'bisnis',
    'suami', 'istri', 'anakku', 'anak-anak', 'mertua', 'ipar', 'ponakan',
    'nikah', 'rumah tangga', 'pemerintah', 'kebijakan', 'ekonomi', 'bansos'
]

NON_INDIVIDUAL_KEYWORDS = [
    'news', 'berita', 'media', 'official', 'bot', 'fess', 'auto', 'base',
    'loker', 'info', 'tv', 'fm', 'radio', 'humas', 'admin', 'store', 'shop'
]

def classify_subject(text, user=''):
    """Classify subject into REMAJA, DEWASA, INDIVIDU_UMUM, or MEDIA_BOT"""
    text = str(text).lower()
    user = str(user).lower()
    
    # Check for non-individual (media/bot)
    for kw in NON_INDIVIDUAL_KEYWORDS:
        if kw in user:
            return 'MEDIA_BOT'
    
    teen_score = sum(1 for w in TEEN_INDICATORS if w in text)
    adult_score = sum(1 for w in ADULT_INDICATORS if w in text)
    
    if teen_score > adult_score and teen_score >= 1:
        return 'REMAJA'
    elif adult_score > teen_score and adult_score >= 1:
        return 'DEWASA'
    else:
        return 'INDIVIDU_UMUM'

def add_subject_classification(df):
    """Add subject_category column to dataframe if not present"""
    if 'subject_category' not in df.columns:
        text_col = 'post_text' if 'post_text' in df.columns else 'Tweet' if 'Tweet' in df.columns else None
        user_col = 'user_id' if 'user_id' in df.columns else 'User' if 'User' in df.columns else None
        
        if text_col:
            df['subject_category'] = df.apply(
                lambda row: classify_subject(
                    row[text_col], 
                    row[user_col] if user_col else ''
                ), 
                axis=1
            )
    return df

# =============================================================================
# SECTION 4: DATA LOADING
# =============================================================================

def detect_paths():
    """Detect output paths"""
    base_paths = ["MentalTrendAI_Output", "data/pipeline_output", "pipeline_output"]
    found_model = "MentalTrendAI_Output/models"
    found_data = "MentalTrendAI_Output/processed_data"
    found_eval = "MentalTrendAI_Output/evaluations"
    found_filter = "data"
    
    for base in base_paths:
        if os.path.exists(os.path.join(base, "models")):
            found_model = os.path.join(base, "models")
        if os.path.exists(os.path.join(base, "processed_data")):
            found_data = os.path.join(base, "processed_data")
        if os.path.exists(os.path.join(base, "evaluations")):
            found_eval = os.path.join(base, "evaluations")
    
    return found_model, found_data, found_eval, found_filter

@st.cache_resource
def load_resources(model_path, data_path, eval_path, filter_path):
    """Load all resources"""
    resources = {'models': {}, 'data': None, 'evaluations': {}, 'filter_reports': {}}
    
    # Load Models
    if os.path.exists(model_path):
        try:
            lstm_path = os.path.join(model_path, 'lstm_model.keras')
            if not os.path.exists(lstm_path):
                lstm_path = os.path.join(model_path, 'lstm_model.h5')
            if os.path.exists(lstm_path):
                resources['models']['lstm_model'] = keras_load_model(lstm_path)
            
            scaler_path = os.path.join(model_path, 'lstm_scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    resources['models']['scaler'] = pickle.load(f)
        except Exception as e:
            st.warning(f"Model loading warning: {e}")
    
    # Load Data - prioritize processed data, add subject classification if needed
    csv_path = os.path.join(data_path, 'full_preprocessed_data.csv')
    if os.path.exists(csv_path):
        resources['data'] = pd.read_csv(csv_path)
        if 'post_created' in resources['data'].columns:
            resources['data']['post_created'] = pd.to_datetime(resources['data']['post_created'])
        # Add subject classification on-the-fly
        resources['data'] = add_subject_classification(resources['data'])
    
    # Load Evaluations
    for eval_file in ['lstm_model_evaluation.json', 'pipeline_summary_report.json']:
        eval_file_path = os.path.join(eval_path, eval_file)
        if os.path.exists(eval_file_path):
            with open(eval_file_path, 'r') as f:
                resources['evaluations'][eval_file.replace('.json', '')] = json.load(f)
    
    # Load Filter Reports
    for report_file in ['mental_health_filter_report.json', 'preprocessing_report.json']:
        report_path = os.path.join(filter_path, report_file)
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                resources['filter_reports'][report_file.replace('.json', '')] = json.load(f)
    
    return resources

# =============================================================================
# SECTION 4: VISUALIZATION FUNCTIONS
# =============================================================================

def create_sentiment_donut(df):
    """Create modern sentiment donut chart"""
    if 'sentiment_category' not in df.columns:
        return go.Figure()
    
    counts = df['sentiment_category'].value_counts()
    colors = {'Negatif': '#F43F5E', 'Positif': '#10B981', 'Neutral': '#6366F1', 'Netral': '#6366F1'}
    
    fig = go.Figure(data=[go.Pie(
        labels=counts.index, values=counts.values, hole=0.7,
        marker=dict(colors=[colors.get(l, '#94A3B8') for l in counts.index]),
        textinfo='percent', textfont=dict(color='#FFFFFF', size=14, family="Plus Jakarta Sans"),
        hoverinfo='label+value'
    )])
    
    fig.add_annotation(
        text=f"<span style='font-size:28px; font-weight:800; color:#FFFFFF'>{counts.sum():,}</span><br><span style='font-size:12px; color:#94A3B8'>POSTS</span>",
        x=0.5, y=0.5, showarrow=False
    )
    
    fig.update_layout(
        showlegend=True, legend=dict(orientation="h", y=-0.1, x=0.5, xanchor='center', font=dict(color='#FFFFFF', size=12)),
        height=320, margin=dict(t=10, b=30, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_mh_category_chart(df):
    """Create mental health category distribution chart"""
    if 'mh_category' not in df.columns:
        return go.Figure()
    
    counts = df['mh_category'].value_counts().head(10)
    colors = px.colors.sequential.Viridis
    
    fig = go.Figure(data=[go.Bar(
        x=counts.values, y=counts.index, orientation='h',
        marker=dict(color=counts.values, colorscale='Viridis'),
        text=counts.values, textposition='outside',
        textfont=dict(color='#FFFFFF', size=12)
    )])
    
    fig.update_layout(
        height=350, margin=dict(t=10, b=10, l=10, r=50),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#E2E8F0')),
        yaxis=dict(showgrid=False, tickfont=dict(color='#FFFFFF', size=11))
    )
    return fig

def create_subject_distribution(df):
    """Create subject demographics distribution"""
    if 'subject_category' not in df.columns:
        return go.Figure()
    
    counts = df['subject_category'].value_counts()
    colors = {'REMAJA': '#F472B6', 'DEWASA': '#38BDF8', 'INDIVIDU_UMUM': '#A78BFA', 
              'INDIVIDU_CAMPURAN': '#FBBF24', 'MEDIA_BOT': '#94A3B8'}
    
    fig = go.Figure(data=[go.Pie(
        labels=counts.index, values=counts.values, hole=0.6,
        marker=dict(colors=[colors.get(l, '#94A3B8') for l in counts.index]),
        textinfo='percent+label', textfont=dict(color='#FFFFFF', size=11)
    )])
    
    fig.update_layout(
        height=300, margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    return fig

def create_time_series_chart(df, models):
    """Create time series sentiment chart"""
    if df.empty or 'post_created' not in df.columns:
        return go.Figure()
    
    df['date'] = pd.to_datetime(df['post_created']).dt.date
    colors = {'Negatif': '#F43F5E', 'Positif': '#10B981', 'Neutral': '#6366F1', 'Netral': '#6366F1'}
    
    fig = go.Figure()
    
    for sentiment in df['sentiment_category'].unique() if 'sentiment_category' in df.columns else []:
        daily = df[df['sentiment_category'] == sentiment].groupby('date').size()
        if len(daily) > 0:
            fig.add_trace(go.Scatter(
                x=daily.index, y=daily.values, mode='lines',
                name=sentiment, line=dict(color=colors.get(sentiment, '#94A3B8'), width=2.5, shape='spline')
            ))
    
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified",
        xaxis=dict(showgrid=False, tickfont=dict(color='#E2E8F0', size=10)),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#E2E8F0')),
        legend=dict(orientation="h", y=1.1, font=dict(color='#FFFFFF', size=11))
    )
    return fig

def create_users_per_day_chart(df, subject_filter='Semua'):
    """Create unique users per day chart"""
    if df.empty:
        return go.Figure()
    
    df_temp = df.copy()
    df_temp['date'] = pd.to_datetime(df_temp['post_created']).dt.date
    
    if subject_filter != 'Semua' and 'subject_category' in df_temp.columns:
        df_temp = df_temp[df_temp['subject_category'] == subject_filter]
    
    if 'user_id' in df_temp.columns:
        daily_users = df_temp.groupby('date')['user_id'].nunique()
    else:
        daily_users = df_temp.groupby('date').size()
    
    fig = go.Figure(data=[go.Scatter(
        x=daily_users.index, y=daily_users.values,
        mode='lines+markers', fill='tozeroy',
        line=dict(color='#818CF8', width=2.5, shape='spline'),
        marker=dict(size=4, color='#818CF8'),
        fillcolor='rgba(129, 140, 248, 0.15)'
    )])
    
    fig.update_layout(
        height=280, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, tickfont=dict(color='#E2E8F0', size=10)),
        yaxis=dict(title=dict(text="Users", font=dict(color='#FFFFFF', size=11)), 
                   showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#E2E8F0'))
    )
    return fig

# =============================================================================
# SECTION 5: MAIN APPLICATION
# =============================================================================

def main():
    # Sidebar
    st.sidebar.markdown("## ⚙️ Settings")
    st.sidebar.markdown("---")
    
    model_path, data_path, eval_path, filter_path = detect_paths()
    
    # Load resources
    resources = load_resources(model_path, data_path, eval_path, filter_path)
    df = resources['data']
    models = resources['models']
    evaluations = resources['evaluations']
    filter_reports = resources['filter_reports']
    
    # Header
    st.markdown("""
    <div class="modern-header">
        <h1>🧠 MentalTrend AI</h1>
        <p>Platform Analisis Kesehatan Mental Berbasis AI dengan Subject Demographics Filter</p>
    </div>
    """, unsafe_allow_html=True)
    
    if df is None:
        st.error("⚠️ Data tidak ditemukan. Jalankan pipeline terlebih dahulu.")
        st.stop()
    
    # Filters Section
    st.markdown('<p class="section-header">🔎 Filter Analisis</p>', unsafe_allow_html=True)
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        time_opt = st.selectbox("📅 Rentang Waktu", ["Semua", "30 Hari", "90 Hari", "1 Tahun"])
    with col_f2:
        mh_categories = ["Semua"] + sorted(df['mh_category'].dropna().unique().tolist()) if 'mh_category' in df.columns else ["Semua"]
        selected_mh = st.selectbox("🏷️ Kategori MH", mh_categories)
    with col_f3:
        demo_options = ["Semua"]
        if 'subject_category' in df.columns:
            demo_options += sorted(df['subject_category'].dropna().unique().tolist())
        selected_demo = st.selectbox("👥 Demografi", demo_options)
    
    # Apply Filters
    df_filt = df.copy()
    if time_opt != "Semua":
        days = {"30 Hari": 30, "90 Hari": 90, "1 Tahun": 365}.get(time_opt, 365)
        df_filt = df_filt[df_filt['post_created'] >= df_filt['post_created'].max() - timedelta(days=days)]
    if selected_mh != "Semua" and 'mh_category' in df_filt.columns:
        df_filt = df_filt[df_filt['mh_category'] == selected_mh]
    if selected_demo != "Semua" and 'subject_category' in df_filt.columns:
        df_filt = df_filt[df_filt['subject_category'] == selected_demo]
    
    # Key Metrics Row
    st.markdown('<p class="section-header">📊 Key Metrics</p>', unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    
    m1.metric("Total Posts", f"{len(df_filt):,}")
    
    unique_users = df_filt['user_id'].nunique() if 'user_id' in df_filt.columns else len(df_filt)
    m2.metric("Unique Users", f"{unique_users:,}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main Dashboard Grid
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown('<p class="section-header">💭 Sentiment Distribution</p>', unsafe_allow_html=True)
        st.plotly_chart(create_sentiment_donut(df_filt), use_container_width=True)
    
    with col2:
        st.markdown('<p class="section-header">📈 Sentiment Time Series</p>', unsafe_allow_html=True)
        st.plotly_chart(create_time_series_chart(df_filt, models), use_container_width=True)
    

    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748B; font-size: 0.85rem;">
        <p>🧠 MentalTrendAI v2.0 | Subject Demographics Integration | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()