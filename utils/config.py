# ============================================
# KONFIGURASI GOOGLE DRIVE
# ============================================
# GANTI DENGAN FILE ID ANDA!
# Cara mendapatkannya: 
# 1. Upload file ke Google Drive
# 2. Share file dengan "Anyone with link"
# 3. Ambil ID dari link: https://drive.google.com/file/d/XXXXXXXXX/view

GOOGLE_DRIVE_CONFIG = {
    # Gunakan ZIP file (lebih cepat dan praktis)
    'zip_file': {
        'id': '1ABC123XYZ',  # <--- GANTI DENGAN ID FILE ZIP ANDA
        'filename': 'models.zip',
        'extract_to': 'models'
    },
    
    # Metadata
    'version': '1.0.0',
    'last_updated': '2024-01-15',
    'total_size_mb': 293,  # Sesuaikan dengan ukuran ZIP Anda
    'required_files': [
        'knn_model_best.pkl',
        'rf_model_best.pkl',
        'xgb_model_best.pkl',
        'scaler.pkl',
        'le_cut.pkl',
        'le_color.pkl',
        'le_clarity.pkl'
    ]
}

# ============================================
# KONFIGURASI APLIKASI (APP_CONFIG)
# ============================================
# TAMBAHKAN INI UNTUK MENGATASI ERROR!

APP_CONFIG = {
    "page_title": "Diamond Price Predictor",
    "page_icon": "💎",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "app_title": "💎 Diamond Price Prediction",
    "app_description": "### Prediksi harga diamond menggunakan 3 algoritma Machine Learning",
    "logo_url": "https://cdn-icons-png.flaticon.com/512/1995/1995570.png",
    "version": "1.0.0"
}

# ============================================
# INFORMASI MODEL (MODEL_INFO)
# ============================================
MODEL_INFO = {
    "K-Nearest Neighbors (KNN)": {
        "R2 Score": 0.9642,
        "MAE": 378.50,
        "RMSE": 756.14,
        "Kecepatan": "Sedang",
        "Kelebihan": "Mudah dipahami, non-parametrik",
        "Kekurangan": "Lambat untuk dataset besar"
    },
    "Random Forest": {
        "R2 Score": 0.9816,
        "MAE": 267.65,
        "RMSE": 540.97,
        "Kecepatan": "Cepat",
        "Kelebihan": "Handal, mengatasi overfitting",
        "Kekurangan": "Kurang interpretable"
    },
    "XGBoost": {
        "R2 Score": 0.9822,
        "MAE": 277.40,
        "RMSE": 532.10,
        "Kecepatan": "Sangat Cepat",
        "Kelebihan": "Akurasi tinggi, efisien",
        "Kekurangan": "Banyak hyperparameter"
    }
}
