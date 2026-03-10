# ============================================
# MODEL DOWNLOADER - Download models from Google Drive
# ============================================

import os
import gdown
import zipfile
import streamlit as st
from pathlib import Path
import time
import tempfile

from .config import GOOGLE_DRIVE_CONFIG

class ModelDownloader:
    """Class untuk mengelola download model dari Google Drive"""
    
    def __init__(self):
        self.models_dir = Path('models')
        self.models_dir.mkdir(exist_ok=True)
        self.config = GOOGLE_DRIVE_CONFIG
    
    def download_zip_and_extract(self):
        """Download file zip dan extract"""
        
        if 'zip_file' not in self.config:
            st.error("❌ ZIP configuration not found in config.py")
            return False
        
        zip_config = self.config['zip_file']
        zip_path = self.models_dir / zip_config.get('filename', 'models.zip')
        
        # Download zip file jika belum ada
        if not zip_path.exists():
            st.info("📦 Downloading model archive...")
            url = f'https://drive.google.com/uc?id={zip_config["id"]}'
            
            try:
                with st.spinner("Downloading models.zip..."):
                    gdown.download(url, str(zip_path), quiet=False)
                st.success("✅ ZIP file downloaded!")
            except Exception as e:
                st.error(f"❌ Error downloading ZIP: {str(e)}")
                return False
        
        # Extract zip
        with st.spinner("📂 Extracting model files..."):
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(zip_config.get('extract_to', 'models'))
                st.success("✅ Models extracted successfully!")
                
                # Hapus file zip setelah extract (opsional)
                # zip_path.unlink()
                return True
                
            except Exception as e:
                st.error(f"❌ Error extracting models: {str(e)}")
                return False
    
    def check_files(self):
        """Cek apakah semua file sudah ada"""
        required_files = self.config.get('required_files', [])
        missing = []
        
        for file in required_files:
            if not (self.models_dir / file).exists():
                missing.append(file)
        
        return missing

def check_and_download_models():
    """Main function untuk cek dan download models"""
    
    downloader = ModelDownloader()
    
    # Cek file yang missing
    missing = downloader.check_files()
    
    if not missing:
        st.success("✅ All model files exist!")
        return True
    
    # Ada file missing, download ZIP
    st.warning(f"⚠️ {len(missing)} model files missing. Downloading from Google Drive...")
    
    # Tampilkan file yang missing
    with st.expander("Missing files"):
        for file in missing:
            st.caption(f"  • {file}")
    
    # Download dan extract ZIP
    success = downloader.download_zip_and_extract()
    
    if success:
        # Verifikasi setelah download
        missing_after = downloader.check_files()
        if not missing_after:
            st.success("✅ All models downloaded and ready!")
            return True
        else:
            st.error(f"❌ Still missing {len(missing_after)} files after extraction")
            return False
    else:
        return False
