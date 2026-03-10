import os
import gdown
import zipfile
import hashlib
import streamlit as st
from pathlib import Path
import time
import tempfile
import shutil
import psutil
import gc

from .config import GOOGLE_DRIVE_CONFIG

class ModelDownloader:
    """Class untuk mengelola download model dari Google Drive"""
    
    def __init__(self):
        self.models_dir = Path('models')
        self.models_dir.mkdir(exist_ok=True)
        self.config = GOOGLE_DRIVE_CONFIG
        
    def log_memory(self, stage=""):
        """Log current memory usage"""
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024 / 1024
        st.sidebar.caption(f"💾 Memory {stage}: {mem:.1f} MB")
        return mem
    
    def check_memory_limit(self):
        """Check if approaching memory limit (1GB)"""
        mem = self.log_memory()
        if mem > 900:  # 900MB sudah mendekati limit
            st.warning("⚠️ Memory usage is high. Consider restarting the app.")
            return False
        return True
    
    def download_file_with_delay(self, file_id, output_path, delay=2):
        """Download file dengan delay untuk menghindari timeout"""
        url = f'https://drive.google.com/uc?id={file_id}'
        
        try:
            # Cek memory sebelum download
            self.check_memory_limit()
            
            st.info(f"⬇️ Downloading {os.path.basename(output_path)}...")
            gdown.download(url, output_path, quiet=False)
            
            # Delay antar download
            st.info(f"⏳ Waiting {delay} seconds before next download...")
            time.sleep(delay)
            
            # Force garbage collection
            gc.collect()
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error downloading: {str(e)}")
            return False
    
    def download_all_files_sequential(self):
        """Download semua file secara berurutan dengan delay"""
        
        files_config = self.config.get('files', {})
        total_files = len(files_config)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        downloaded = 0
        for idx, (filename, file_id) in enumerate(files_config.items()):
            status_text.text(f"Downloading {filename}... ({idx+1}/{total_files})")
            
            output_path = self.models_dir / filename
            
            if not output_path.exists():
                # Download dengan delay 3 detik
                if self.download_file_with_delay(file_id, str(output_path), delay=3):
                    downloaded += 1
            else:
                st.info(f"⏭️ {filename} already exists")
                downloaded += 1
            
            progress_bar.progress((idx + 1) / total_files)
        
        status_text.text("Download complete!")
        progress_bar.empty()
        
        return downloaded == total_files
    
    def check_missing_files(self):
        """Cek file yang missing"""
        required_files = self.config.get('required_files', [])
        existing = []
        missing = []
        
        for f in required_files:
            if (self.models_dir / f).exists():
                size = (self.models_dir / f).stat().st_size / (1024*1024)
                existing.append(f"{f} ({size:.1f}MB)")
            else:
                missing.append(f)
        
        return existing, missing


def check_and_download_models():
    """Main function dengan memory monitoring"""
    
    downloader = ModelDownloader()
    
    # Cek memory awal
    downloader.log_memory("start")
    
    # Cek file yang ada
    existing, missing = downloader.check_missing_files()
    
    # Tampilkan status di sidebar
    with st.sidebar:
        with st.expander("📦 Model Status", expanded=False):
            if existing:
                st.success(f"✅ {len(existing)} models ready")
                for f in existing[:3]:
                    st.caption(f"  • {f}")
            
            if missing:
                st.warning(f"⚠️ {len(missing)} models need download")
                for f in missing[:3]:
                    st.caption(f"  • {f}")
    
    # Jika ada file missing, download
    if missing:
        st.warning("⚠️ Model files not found. Starting sequential download...")
        st.info("📊 This may take 2-3 minutes. Please wait...")
        
        success = downloader.download_all_files_sequential()
        
        if success:
            st.success("✅ All models downloaded successfully!")
            downloader.log_memory("after download")
            time.sleep(2)
            st.rerun()
        else:
            st.error("❌ Download failed. Please try again.")
            return False
    
    return True
