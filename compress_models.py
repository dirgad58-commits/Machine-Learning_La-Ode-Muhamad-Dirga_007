"""
Script untuk mengompres model agar ukurannya lebih kecil
Jalankan di lokal sebelum upload ke Google Drive
"""

import joblib
import gzip
import pickle
import os

def compress_model(input_path, output_path):
    """Compress model using gzip"""
    
    # Load model
    print(f"Loading {input_path}...")
    model = joblib.load(input_path)
    
    # Compress and save
    print(f"Compressing to {output_path}...")
    with gzip.open(output_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Show size comparison
    original_size = os.path.getsize(input_path) / (1024*1024)
    compressed_size = os.path.getsize(output_path) / (1024*1024)
    
    print(f"✅ Compressed: {original_size:.1f}MB -> {compressed_size:.1f}MB")
    print(f"   Reduction: {(1 - compressed_size/original_size)*100:.1f}%")
    
    return compressed_size

def load_compressed_model(path):
    """Load compressed model"""
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    # Compress all models
    models = [
        'xgb_model_best.pkl',
        'rf_model_best.pkl',
        'knn_model_best.pkl',
        'scaler.pkl',
        'le_cut.pkl',
        'le_color.pkl',
        'le_clarity.pkl'
    ]
    
    for model in models:
        if os.path.exists(model):
            output = model.replace('.pkl', '.pkl.gz')
            compress_model(model, output)
        else:
            print(f"⚠️ {model} not found")
