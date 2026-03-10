import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import sys

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
# FUNGSI LOAD MODELS (DENGAN AUTO-DOWNLOAD)
# ============================================
@st.cache_resource(ttl=3600)  # Cache selama 1 jam
def load_models():
    """Load semua model dan encoders dengan auto-download jika perlu"""
    
    # Cek dan download models jika diperlukan
    if not check_and_download_models():
        st.error("❌ Gagal mendownload model. Silakan coba lagi nanti.")
        st.stop()
    
    # Load models jika semua file sudah ada
    try:
        with st.spinner("🔄 Loading models..."):
            
            # Load scaler
            scaler = joblib.load('models/scaler.pkl')
            
            # Load label encoders
            le_cut = joblib.load('models/le_cut.pkl')
            le_color = joblib.load('models/le_color.pkl')
            le_clarity = joblib.load('models/le_clarity.pkl')
            
            # Load models
            knn_model = joblib.load('models/knn_model_best.pkl')
            rf_model = joblib.load('models/rf_model_best.pkl')
            xgb_model = joblib.load('models/xgb_model_best.pkl')
            
            # Hitung total size
            downloader = ModelDownloader()
            _, total_size = downloader.get_model_size_info()
            
            st.sidebar.success(f"✅ Models loaded! ({(total_size / (1024**2)):.1f} MB)")
            
            return {
                'scaler': scaler,
                'le_cut': le_cut,
                'le_color': le_color,
                'le_clarity': le_clarity,
                'knn': knn_model,
                'rf': rf_model,
                'xgb': xgb_model,
                'version': APP_CONFIG['version']
            }
            
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.error("Please check if all model files exist and are not corrupted.")
        return None

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
    if model_name == "K-Nearest Neighbors (KNN)":
        prediction = models['knn'].predict(features_scaled)
    elif model_name == "Random Forest":
        prediction = models['rf'].predict(features_scaled)
    elif model_name == "XGBoost":
        prediction = models['xgb'].predict(features_scaled)
    else:
        prediction = None
    
    return prediction[0] if prediction is not None else None

# ============================================
# MAIN APP
# ============================================
def main():
    # Load models (auto-download jika perlu)
    models = load_models()
    if models is None:
        st.stop()
    
    # Simpan models di session state
    st.session_state['models'] = models
    
    # Sidebar untuk input
    with st.sidebar:
        st.image(APP_CONFIG["logo_url"], width=100)
        st.title(APP_CONFIG["app_title"])
        st.caption(f"Version: {models['version']}")
        st.markdown("---")
        
        # Input parameters
        st.header("📊 Input Parameter Diamond")
        
        # Numerical inputs in columns
        col1, col2 = st.columns(2)
        with col1:
            carat = st.number_input(
                "Carat Weight", 
                min_value=0.2, 
                max_value=5.0, 
                value=1.0, 
                step=0.01,
                help="Berat diamond dalam carat (0.2 - 5.0)"
            )
            
            depth = st.number_input(
                "Depth (%)", 
                min_value=40.0, 
                max_value=80.0, 
                value=61.5, 
                step=0.1,
                help="Persentase depth (40% - 80%)"
            )
            
            x = st.number_input(
                "Length (mm)", 
                min_value=0.0, 
                max_value=11.0, 
                value=5.0, 
                step=0.1,
                help="Panjang diamond dalam mm"
            )
        
        with col2:
            table = st.number_input(
                "Table (%)", 
                min_value=40.0, 
                max_value=95.0, 
                value=57.0, 
                step=0.1,
                help="Persentase table (40% - 95%)"
            )
            
            y = st.number_input(
                "Width (mm)", 
                min_value=0.0, 
                max_value=60.0, 
                value=5.0, 
                step=0.1,
                help="Lebar diamond dalam mm"
            )
            
            z = st.number_input(
                "Depth (mm)", 
                min_value=0.0, 
                max_value=32.0, 
                value=3.5, 
                step=0.1,
                help="Kedalaman diamond dalam mm"
            )
        
        st.markdown("---")
        
        # Categorical inputs
        cut = st.selectbox(
            "Cut Quality",
            options=models['le_cut'].classes_,
            help="Kualitas potongan diamond"
        )
        
        color = st.selectbox(
            "Color Grade",
            options=models['le_color'].classes_,
            help="Grade warna diamond (D = terbaik, J = terendah)"
        )
        
        clarity = st.selectbox(
            "Clarity Grade",
            options=models['le_clarity'].classes_,
            help="Grade kejernihan diamond"
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
    # MAIN CONTENT
    # ============================================
    st.title(APP_CONFIG["app_title"])
    st.markdown(APP_CONFIG["app_description"])
    
    # Tabs untuk berbagai fitur
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Prediksi", "📊 Perbandingan Model", "ℹ️ Informasi", "📚 Dataset"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📋 Ringkasan Input")
            
            # Tampilkan input dalam bentuk metrics
            metrics_cols = st.columns(3)
            with metrics_cols[0]:
                st.metric("Carat", f"{carat:.2f}")
                st.metric("Depth", f"{depth:.1f}%")
                st.metric("Table", f"{table:.1f}%")
            
            with metrics_cols[1]:
                st.metric("Cut", cut)
                st.metric("Color", color)
                st.metric("Clarity", clarity)
            
            with metrics_cols[2]:
                st.metric("Length (x)", f"{x:.2f} mm")
                st.metric("Width (y)", f"{y:.2f} mm")
                st.metric("Depth (z)", f"{z:.2f} mm")
        
        with col2:
            st.subheader("🎯 Model Terpilih")
            st.info(f"**{model_choice}**")
            
            # Tampilkan informasi singkat model
            model_info = MODEL_INFO[model_choice]
            st.metric("R2 Score", f"{model_info['R2 Score']:.4f}")
            st.metric("MAE", f"${model_info['MAE']:,.2f}")
        
        st.markdown("---")
        
        # Prediksi
        if predict_button:
            with st.spinner("Menghitung prediksi..."):
                # Buat dataframe fitur
                features_df = create_feature_dataframe(
                    carat, cut, color, clarity, depth, table, x, y, z
                )
                
                if features_df is not None:
                    # Prediksi dengan model yang dipilih
                    prediction = predict_price(features_df, model_choice)
                    
                    if prediction is not None:
                        # Tampilkan hasil prediksi
                        st.subheader("💰 Hasil Prediksi Harga")
                        
                        result_cols = st.columns([1, 2, 1])
                        with result_cols[1]:
                            # Buat gauge chart
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number+delta",
                                value=prediction,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "Predicted Price (USD)"},
                                delta={'reference': 4000},
                                gauge={
                                    'axis': {'range': [None, 20000]},
                                    'bar': {'color': "#1f77b4"},
                                    'steps': [
                                        {'range': [0, 1000], 'color': "lightgray"},
                                        {'range': [1000, 5000], 'color': "gray"},
                                        {'range': [5000, 10000], 'color': "darkgray"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 4900
                                    }
                                }
                            ))
                            
                            fig.update_layout(
                                height=300,
                                margin=dict(l=20, r=20, t=50, b=20),
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Tampilkan harga dalam format mata uang
                        st.markdown(
                            f"""
                            <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px;'>
                                <h2 style='color: white; margin: 0;'>${prediction:,.2f}</h2>
                                <p style='color: rgba(255,255,255,0.8); margin: 0;'>USD</p>
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
        
        # Tampilkan history prediksi
        if 'history' in st.session_state and st.session_state.history:
            st.markdown("---")
            st.subheader("📜 History Prediksi")
            
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(history_df, use_container_width=True)
    
    with tab2:
        st.subheader("📊 Perbandingan Kinerja Model")
        
        # Buat dataframe untuk perbandingan
        comparison_df = pd.DataFrame({
            'Model': list(MODEL_INFO.keys()),
            'R2 Score': [info['R2 Score'] for info in MODEL_INFO.values()],
            'MAE ($)': [info['MAE'] for info in MODEL_INFO.values()],
            'RMSE ($)': [info['RMSE'] for info in MODEL_INFO.values()]
        })
        
        # Tampilkan tabel perbandingan
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Visualisasi perbandingan
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart R2 Score
            fig_r2 = px.bar(
                comparison_df, 
                x='Model', 
                y='R2 Score',
                title='Perbandingan R2 Score',
                color='Model',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_r2.update_layout(showlegend=False)
            st.plotly_chart(fig_r2, use_container_width=True)
        
        with col2:
            # Bar chart MAE
            fig_mae = px.bar(
                comparison_df, 
                x='Model', 
                y='MAE ($)',
                title='Perbandingan MAE (Mean Absolute Error)',
                color='Model',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_mae.update_layout(showlegend=False)
            st.plotly_chart(fig_mae, use_container_width=True)
        
        # Tampilkan detail setiap model
        st.subheader("📋 Detail Model")
        
        for model_name, info in MODEL_INFO.items():
            with st.expander(f"**{model_name}**"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R2 Score", f"{info['R2 Score']:.4f}")
                with col2:
                    st.metric("MAE", f"${info['MAE']:,.2f}")
                with col3:
                    st.metric("RMSE", f"${info['RMSE']:,.2f}")
                
                st.markdown(f"**Kecepatan:** {info['Kecepatan']}")
                st.markdown(f"**Kelebihan:** {info['Kelebihan']}")
                st.markdown(f"**Kekurangan:** {info['Kekurangan']}")
    
    with tab3:
        st.subheader("ℹ️ Informasi Aplikasi")
        
        st.markdown("""
        ### **Tentang Aplikasi**
        Aplikasi ini menggunakan 3 algoritma Machine Learning untuk memprediksi harga diamond berdasarkan karakteristiknya:
        
        - **K-Nearest Neighbors (KNN)**: Algoritma berbasis jarak yang mencari kemiripan dengan data training
        - **Random Forest**: Ensemble method yang menggunakan banyak decision trees
        - **XGBoost**: Algoritma gradient boosting yang sangat akurat dan efisien
        
        ### **Fitur yang Digunakan**
        1. **Carat**: Berat diamond (1 carat = 200 mg)
        2. **Cut**: Kualitas potongan (Fair, Good, Very Good, Premium, Ideal)
        3. **Color**: Warna diamond (D = terbaik, J = terendah)
        4. **Clarity**: Kejernihan (I1 = terendah, IF = terbaik)
        5. **Depth**: Persentase kedalaman
        6. **Table**: Persentase table
        7. **Dimensions**: Panjang (x), Lebar (y), Kedalaman (z) dalam mm
        
        ### **Performa Model**
        - ✅ **XGBoost** adalah model terbaik dengan R2 Score 0.9822
        - ✅ **Random Forest** memiliki MAE terendah ($267.65)
        - ✅ Semua model memiliki akurasi di atas 96%
        
        ### **Cara Penggunaan**
        1. Masukkan parameter diamond di sidebar
        2. Pilih algoritma yang diinginkan
        3. Klik tombol "Prediksi Harga"
        4. Lihat hasil prediksi dan history
        """)
    
    with tab4:
        st.subheader("📚 Preview Dataset")
        
        # Load sample data
        try:
            df_sample = pd.read_csv('diamonds.csv').head(100)
            st.dataframe(df_sample, use_container_width=True)
            
            # Statistik deskriptif
            st.subheader("📊 Statistik Deskriptif")
            st.dataframe(df_sample.describe(), use_container_width=True)
            
            # Distribusi harga
            fig_dist = px.histogram(
                df_sample, 
                x='price', 
                nbins=50,
                title='Distribusi Harga Diamond (Sample 100 data)',
                labels={'price': 'Price (USD)'}
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
        except Exception as e:
            st.warning("Dataset tidak ditemukan. Tampilkan hanya sample data.")
            st.info("Dataset original dapat diunduh dari Kaggle.")

# ============================================
# RUN APLIKASI
# ============================================
if __name__ == '__main__':
    main()
