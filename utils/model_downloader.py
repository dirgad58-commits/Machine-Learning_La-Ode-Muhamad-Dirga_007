import os
import gdown
import zipfile
import streamlit as st
from pathlib import Path
import time

class ModelDownloader:
    def __init__(self):
        self.models_dir = Path('models')
        self.models_dir.mkdir(exist_ok=True)
        self.config = GOOGLE_DRIVE_CONFIG  # Import dari config.py
    
    def download_zip_and_extract(self):
        """Download ZIP file dan extract"""
        
        zip_config = self.config.get('zip_file', {})
        if not zip_config:
            st.error("❌ ZIP configuration not found")
            return False
        
        zip_path = self.models_dir / zip_config.get('filename', 'models.zip')
        
        # Download ZIP hanya jika belum ada
        if not zip_path.exists():
            st.info("📦 Downloading model archive...")
            url = f'https://drive.google.com/uc?id={zip_config["id"]}'
            
            with st.spinner("Downloading models.zip..."):
                gdown.download(url, str(zip_path), quiet=False)
        
        # Extract ZIP
        with st.spinner("📂 Extracting model files..."):
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(zip_config.get('extract_to', 'models'))
                st.success("✅ Models extracted successfully!")
                
                # Hapus ZIP setelah extract (opsional)
                # zip_path.unlink()
                return True
                
            except Exception as e:
                st.error(f"❌ Error extracting models: {str(e)}")
                return False
    
    def check_files(self):
        """Cek apakah semua file sudah ada"""
        required = self.config.get('required_files', [])
        missing = []
        
        for file in required:
            if not (self.models_dir / file).exists():
                missing.append(file)
        
        return missing

def check_and_download_models():
    """Main function"""
    downloader = ModelDownloader()
    
    # Cek file yang missing
    missing = downloader.check_files()
    
    if missing:
        st.warning(f"⚠️ {len(missing)} model files missing")
        return downloader.download_zip_and_extract()
    else:
        st.success("✅ All model files exist!")
        return True
