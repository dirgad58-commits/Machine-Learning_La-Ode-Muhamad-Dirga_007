# ============================================
# KONFIGURASI GOOGLE DRIVE
# ============================================
# GANTI DENGAN FILE ID ANDA!
# Cara mendapatkannya: 
# 1. Upload file ke Google Drive
# 2. Share file dengan "Anyone with link"
# 3. Ambil ID dari link: https://drive.google.com/file/d/XXXXXXXXX/view

GOOGLE_DRIVE_CONFIG = {
    # Jika upload file satu per satu
    'files': {
        'xgb_model_best.pkl': '1Di-9Ztl29PEiTieEN-rMsbHz_thxOO8D',  # Ganti dengan ID file XGBoost
        'rf_model_best.pkl': '1nJAHUhAjOi_7n5V1Wwp8T0Jchd2q85UP',    # Ganti dengan ID file Random Forest
        'knn_model_best.pkl': '1cg-3XlNXqL9pFNtRXR39fzZQpTJUs7Ks',   # Ganti dengan ID file KNN
        'scaler.pkl': '11gkpUXg42MF8WCB1iNFN_rD1kEu34hmjO',           # Ganti dengan ID file Scaler
        'le_cut.pkl': '1Ry1Rs4Ajbduo5RD_7YGCeGwvemBlspJ_',           # Ganti dengan ID file LabelEncoder Cut
        'le_color.pkl': '1cLjXClULTcQc1wTe2jZ64RJLreUQMuzC',         # Ganti dengan ID file LabelEncoder Color
        'le_clarity.pkl': '1V78kj-AA8eV_IWv1h1zGo5m5s42iw0MY'        # Ganti dengan ID file LabelEncoder Clarity
    },
    
    # Atau jika upload dalam 1 file zip
    'zip_file': {
        'id': '1ZIPFILEID123456789',  # Ganti dengan ID file ZIP
        'filename': 'models.zip',
        'extract_to': 'models'
    },
    
    # Metadata
    'version': '1.0.0',
    'last_updated': '2024-01-15',
    'total_size_mb': 200,  # Estimasi total size dalam MB
    'required_files': [
        'scaler.pkl',
        'le_cut.pkl',
        'le_color.pkl',
        'le_clarity.pkl',
        'knn_model_best.pkl',
        'rf_model_best.pkl',
        'xgb_model_best.pkl'
    ]
}

# ============================================
# KONFIGURASI APLIKASI
# ============================================
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
# INFORMASI MODEL
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
