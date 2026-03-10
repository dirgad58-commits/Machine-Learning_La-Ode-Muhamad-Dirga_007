# ============================================
# DIAMOND PRICE PREDICTION - STREAMLIT APP
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import sys
import gc
import time
import traceback

# Tambahkan path untuk import modul utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.model_downloader import ModelDownloader, check_and_download_models
from utils.config import APP_CONFIG, MODEL_INFO

# ============================================
# KONFIGURASI HALAMAN
# ============================================
st.set_page_config(
    page_title=APP_CONFIG["page_title"],
    page_icon=APP_CONFIG["page_icon"],
    layout=APP_CONFIG["layout"],
    initial_sidebar_state=APP_CONFIG["initial_sidebar_state"]
)

# ============================================
# FUNGSI MEMORY MONITORING
# ============================================
def log_memory(stage=""):
    """Monitor memory usage"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024 / 1024
        st.sidebar.caption(f"💾 RAM {stage}: {mem:.1f} MB")
        return mem
    except:
        return 0

def force_gc():
    """Force garbage collection"""
    gc.collect()
    time.sleep(0.5)

# ============================================
# LOAD MODELS - SATU PER SATU
# ============================================
@st.cache_resource(ttl=3600)
def load_models():
    """Load models satu per satu untuk menghemat memory"""
    
    # Log memory awal
    log_memory("start")
    
    # Download models jika perlu
    with st.spinner("📥 Checking model files..."):
        if not check_and_download_models():
            st.error("❌ Failed to download models")
            st.stop()
    
    models = {}
    
    # 1. Load scaler (paling ringan)
    with st.spinner("🔄 Loading scaler..."):
        try:
            models['scaler'] = joblib.load('models/scaler.pkl')
            log_memory("after scaler")
            force_gc()
        except Exception as e:
            st.error(f"❌ Error loading scaler: {str(e)}")
            st.stop()
    
    # 2. Load label encoders
    with st.spinner("🔄 Loading encoders..."):
        try:
            models['le_cut'] = joblib.load('models/le_cut.pkl')
            models['le_color'] = joblib.load('models/le_color.pkl')
            models['le_clarity'] = joblib.load('models/le_clarity.pkl')
            log_memory("after encoders")
            force_gc()
        except Exception as e:
            st.error(f"❌ Error loading encoders: {str(e)}")
            st.stop()
    
    # 3. Load KNN (paling ringan di antara model)
    with st.spinner("🔄 Loading KNN model..."):
        try:
            models['knn'] = joblib.load('models/knn_model_best.pkl')
            log_memory("after KNN")
            force_gc()
        except Exception as e:
            st.error(f"❌ Error loading KNN model: {str(e)}")
            st.stop()
    
    # 4. Load Random Forest (sedang)
    with st.spinner("🔄 Loading Random Forest model..."):
        try:
            models['rf'] = joblib.load('models/rf_model_best.pkl')
            log_memory("after RF")
            force_gc()
        except Exception as e:
            st.error(f"❌ Error loading Random Forest model: {str(e)}")
            st.stop()
    
    # 5. Load XGBoost (paling berat)
    with st.spinner("🔄 Loading XGBoost model..."):
        try:
            models['xgb'] = joblib.load('models/xgb_model_best.pkl')
            log_memory("after XGB")
            force_gc()
        except Exception as e:
            st.error(f"❌ Error loading XGBoost model: {str(e)}")
            st.stop()
    
    # Final memory
    log_memory("final")
    
    return models

# ============================================
# FUNGSI UTILITY
# ============================================
def create_feature_dataframe(carat, cut, color, clarity, depth, table, x, y, z):
    """Membuat dataframe fitur untuk prediksi"""
    models = st.session_state.get('models', None)
    if models is None:
        return None
    
    # Encode categorical features
    cut_encoded = models['le_cut'].transform([cut])[0]
    color_encoded = models['le_color'].transform([color])[0]
    clarity_encoded = models['le_clarity'].transform([clarity])[0]
    
    # Buat dataframe
    data = {
        'carat': [carat],
        'cut_encoded': [cut_encoded],
        'color_encoded': [color_encoded],
        'clarity_encoded': [clarity_encoded],
        'depth': [depth],
        'table': [table],
        'x': [x],
        'y': [y],
        'z': [z]
    }
    
    df = pd.DataFrame(data)
    return df

def predict_price(features_df, model_name):
    """Melakukan prediksi harga"""
    models = st.session_state.get('models', None)
    if models is None:
        return None
    
    # Scale features
    features_scaled = models['scaler'].transform(features_df)
    
    # Prediksi berdasarkan model yang dipilih
    try:
        if model_name == "K-Nearest Neighbors (KNN)":
            prediction = models['knn'].predict(features_scaled)
        elif model_name == "Random Forest":
            prediction = models['rf'].predict(features_df)  # RF tidak perlu scaling
        elif model_name == "XGBoost":
            prediction = models['xgb'].predict(features_df)  # XGBoost tidak perlu scaling
        else:
            prediction = None
        
        return prediction[0] if prediction is not None else None
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        return None

def format_price(price):
    """Format harga ke dalam mata uang"""
    return f"${price:,.2f}"

# ============================================
# MAIN APP
# ============================================
def main():
    # Title and description
    st.title(APP_CONFIG["app_title"])
    st.markdown(APP_CONFIG["app_description"])
    
    # Load models
    with st.spinner("🚀 Initializing application..."):
        models = load_models()
        if models is None:
            st.stop()
    
    # Simpan models di session state
    st.session_state['models'] = models
    
    # Force GC setelah load
    force_gc()
    
    # ============================================
    # SIDEBAR - INPUT DATA (USER INTERACTION)
    # ============================================
    with st.sidebar:
        st.image(APP_CONFIG["logo_url"], width=100)
        st.title(APP_CONFIG["app_title"])
        st.caption(f"Version: {models.get('version', APP_CONFIG['version'])}")
        
        # Tampilkan memory usage
        log_memory("sidebar")
        
        st.markdown("---")
        
        # Input parameters
        st.header("📊 Input Parameter Diamond")
        
        # Numerical inputs in columns
        col1, col2 = st.columns(2)
        
        with col1:
            carat = st.number_input(
                "⚖️ Carat Weight", 
                min_value=0.2, 
                max_value=5.0, 
                value=1.0, 
                step=0.01,
                help="Berat diamond dalam carat (0.2 - 5.0)"
            )
            
            depth = st.number_input(
                "📏 Depth (%)", 
                min_value=40.0, 
                max_value=80.0, 
                value=61.5, 
                step=0.1,
                help="Persentase depth (40% - 80%)"
            )
            
            x = st.number_input(
                "📐 Length (mm)", 
                min_value=0.0, 
                max_value=11.0, 
                value=5.0, 
                step=0.1,
                help="Panjang diamond dalam mm"
            )
        
        with col2:
            table = st.number_input(
                "📊 Table (%)", 
                min_value=40.0, 
                max_value=95.0, 
                value=57.0, 
                step=0.1,
                help="Persentase table (40% - 95%)"
            )
            
            y = st.number_input(
                "📐 Width (mm)", 
                min_value=0.0, 
                max_value=60.0, 
                value=5.0, 
                step=0.1,
                help="Lebar diamond dalam mm"
            )
            
            z = st.number_input(
                "📐 Depth (mm)", 
                min_value=0.0, 
                max_value=32.0, 
                value=3.5, 
                step=0.1,
                help="Kedalaman diamond dalam mm"
            )
        
        st.markdown("---")
        
        # Categorical inputs
        cut = st.selectbox(
            "💎 Cut Quality",
            options=models['le_cut'].classes_,
            help="Kualitas potongan diamond (Fair, Good, Very Good, Premium, Ideal)"
        )
        
        color = st.selectbox(
            "🎨 Color Grade",
            options=models['le_color'].classes_,
            help="Grade warna diamond (D = terbaik, J = terendah)"
        )
        
        clarity = st.selectbox(
            "✨ Clarity Grade",
            options=models['le_clarity'].classes_,
            help="Grade kejernihan diamond (I1 = terendah, IF = terbaik)"
        )
        
        st.markdown("---")
        
        # Model selection
        st.header("🤖 Pilih Model")
        model_choice = st.radio(
            "Algoritma Prediksi",
            options=["K-Nearest Neighbors (KNN)", "Random Forest", "XGBoost"],
            index=2,  # Default XGBoost
            help="Pilih algoritma machine learning untuk prediksi"
        )
        
        st.markdown("---")
        
        # Predict button
        predict_button = st.button("🔮 Prediksi Harga", type="primary", use_container_width=True)
        
        if st.button("🔄 Reset", use_container_width=True):
            st.rerun()
    
    # ============================================
    # MAIN CONTENT - HASIL PREDIKSI
    # ============================================
    
    # Ringkasan Input
    st.subheader("📋 Ringkasan Input")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Carat", f"{carat:.2f}")
        st.metric("Depth", f"{depth:.1f}%")
        st.metric("Table", f"{table:.1f}%")
    
    with col2:
        st.metric("Cut", cut)
        st.metric("Color", color)
        st.metric("Clarity", clarity)
    
    with col3:
        st.metric("Length (x)", f"{x:.2f} mm")
        st.metric("Width (y)", f"{y:.2f} mm")
        st.metric("Depth (z)", f"{z:.2f} mm")
    
    st.markdown("---")
    
    # Model terpilih
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎯 Model Terpilih")
        st.info(f"**{model_choice}**")
        
        # Tampilkan informasi singkat model
        model_info = MODEL_INFO[model_choice]
        st.metric("R2 Score", f"{model_info['R2 Score']:.4f}")
        st.metric("MAE", f"${model_info['MAE']:,.2f}")
    
    # Prediksi
    with col2:
        if predict_button:
            with st.spinner("🔮 Menghitung prediksi..."):
                # Buat dataframe fitur
                features_df = create_feature_dataframe(
                    carat, cut, color, clarity, depth, table, x, y, z
                )
                
                if features_df is not None:
                    # Prediksi dengan model yang dipilih
                    prediction = predict_price(features_df, model_choice)
                    
                    if prediction is not None:
                        st.subheader("💰 Hasil Prediksi")
                        
                        # Tampilkan harga dalam format besar
                        st.markdown(
                            f"""
                            <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px;'>
                                <h1 style='color: white; margin: 0; font-size: 48px;'>{format_price(prediction)}</h1>
                                <p style='color: rgba(255,255,255,0.9); margin: 5px 0 0 0;'>USD</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Simpan history prediksi di session state
                        if 'history' not in st.session_state:
                            st.session_state.history = []
                        
                        st.session_state.history.append({
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'model': model_choice,
                            'carat': carat,
                            'cut': cut,
                            'color': color,
                            'clarity': clarity,
                            'prediction': prediction
                        })
    
    st.markdown("---")
    
    # Gauge chart untuk visualisasi
    if predict_button and prediction is not None:
        st.subheader("📊 Visualisasi Harga")
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Harga Diamond (USD)"},
            gauge={
                'axis': {'range': [0, 20000]},
                'bar': {'color': "#1f77b4"},
                'steps': [
                    {'range': [0, 1000], 'color': "#e6f3ff"},
                    {'range': [1000, 5000], 'color': "#b3d9ff"},
                    {'range': [5000, 10000], 'color': "#80bfff"},
                    {'range': [10000, 15000], 'color': "#4da6ff"},
                    {'range': [15000, 20000], 'color': "#1a8cff"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 15000
                }
            }
        ))
        
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # History prediksi
    if 'history' in st.session_state and st.session_state.history:
        st.subheader("📜 History Prediksi")
        
        history_df = pd.DataFrame(st.session_state.history)
        
        # Format harga
        history_df['prediction'] = history_df['prediction'].apply(lambda x: f"${x:,.2f}")
        
        st.dataframe(
            history_df,
            use_container_width=True,
            column_config={
                "timestamp": "Waktu",
                "model": "Model",
                "carat": "Carat",
                "cut": "Cut",
                "color": "Color",
                "clarity": "Clarity",
                "prediction": "Harga"
            }
        )
    
    st.markdown("---")
    
    # Informasi tambahan
    with st.expander("ℹ️ Informasi Aplikasi"):
        st.markdown("""
        ### **Tentang Aplikasi**
        Aplikasi ini menggunakan 3 algoritma Machine Learning untuk memprediksi harga diamond:
        
        - **K-Nearest Neighbors (KNN)**: Algoritma berbasis jarak
        - **Random Forest**: Ensemble method dengan banyak decision trees
        - **XGBoost**: Gradient boosting dengan akurasi tinggi
        
        ### **Fitur Input**
        - **Carat**: Berat diamond (0.2 - 5.0 carat)
        - **Cut**: Kualitas potongan (Fair, Good, Very Good, Premium, Ideal)
        - **Color**: Warna diamond (D = terbaik, J = terendah)
        - **Clarity**: Kejernihan (I1 = terendah, IF = terbaik)
        - **Depth**: Persentase kedalaman (40% - 80%)
        - **Table**: Persentase table (40% - 95%)
        - **Dimensions**: Panjang (x), Lebar (y), Kedalaman (z) dalam mm
        """)

# ============================================
# RUN APLIKASI
# ============================================
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.code(traceback.format_exc())
