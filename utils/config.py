# ============================================
# KONFIGURASI GOOGLE DRIVE - PAKAI FILE ZIP
# ============================================

GOOGLE_DRIVE_CONFIG = {
    # Gunakan ZIP file (lebih cepat dan praktis)
    'zip_file': {
        'id': '12HsXFpUdTHgC9eruaI_8jwsLjYWlpCS4',  # <--- GANTI DENGAN FILE ID ANDA
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
