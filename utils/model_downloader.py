import os
import gdown
import zipfile
import hashlib
import streamlit as st
from pathlib import Path
import time
import tempfile
import shutil

from .config import GOOGLE_DRIVE_CONFIG

class ModelDownloader:
    """Class untuk mengelola download model dari Google Drive"""
    
    def __init__(self):
        self.models_dir = Path('models')
        self.models_dir.mkdir(exist_ok=True)
        self.config = GOOGLE_DRIVE_CONFIG
        
    def calculate_md5(self, filepath):
        """Hitung MD5 hash file untuk verifikasi"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def verify_file(self, filepath, expected_size=None):
        """Verifikasi file download"""
        if not os.path.exists(filepath):
            return False
        
        # Cek ukuran file
        if expected_size:
            actual_size = os.path.getsize(filepath)
            if abs(actual_size - expected_size) > 1024:  # Toleransi 1KB
                return False
        
        return True
    
    def check_disk_space(self, required_space_mb=None):
        """Cek ketersediaan disk space"""
        if required_space_mb is None:
            required_space_mb = self.config.get('total_size_mb', 500)
        
        total, used, free = shutil.disk_usage('.')
        free_mb = free / (1024**2)
        
        if free_mb < required_space_mb:
            st.error(f"❌ Insufficient disk space! Need {required_space_mb}MB, have {free_mb:.0f}MB")
            return False
        return True
    
    def download_file(self, file_id, output_path, expected_size=None, show_progress=True):
        """Download file dari Google Drive dengan progress bar"""
        
        # Buat URL download
        url = f'https://drive.google.com/uc?id={file_id}'
        
        # Download dengan progress bar
        try:
            if show_progress:
                with st.spinner(f"Downloading {os.path.basename(output_path)}..."):
                    # Gunakan gdown dengan output ke temporary file
                    temp_file = tempfile.NamedTemporaryFile(delete=False)
                    gdown.download(url, temp_file.name, quiet=True)
                    
                    # Verifikasi file
                    if self.verify_file(temp_file.name, expected_size):
                        # Pindahkan ke lokasi tujuan
                        os.replace(temp_file.name, output_path)
                        st.success(f"✅ {os.path.basename(output_path)} downloaded successfully!")
                        return True
                    else:
                        st.error(f"❌ Download failed: {os.path.basename(output_path)} corrupted")
                        os.unlink(temp_file.name)
                        return False
            else:
                # Download tanpa progress bar
                gdown.download(url, output_path, quiet=True)
                return True
                    
        except Exception as e:
            if show_progress:
                st.error(f"❌ Error downloading {os.path.basename(output_path)}: {str(e)}")
            return False
    
    def download_all_files(self):
        """Download semua file model satu per satu"""
        
        if not self.check_disk_space():
            return False
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        files_config = self.config.get('files', {})
        total_files = len(files_config)
        downloaded_files = []
        
        for idx, (filename, file_id) in enumerate(files_config.items()):
            status_text.text(f"Downloading {filename}... ({idx + 1}/{total_files})")
            
            output_path = self.models_dir / filename
            
            if not output_path.exists():
                if self.download_file(file_id, str(output_path), show_progress=False):
                    downloaded_files.append(filename)
            else:
                st.info(f"⏭️ {filename} already exists, skipping...")
                downloaded_files.append(filename)
            
            # Update progress
            progress_bar.progress((idx + 1) / total_files)
            time.sleep(0.5)  # Sedikit delay untuk UX
        
        status_text.text("Download complete!")
        progress_bar.empty()
        
        return len(downloaded_files) == total_files
    
    def download_zip_and_extract(self):
        """Download file zip dan extract"""
        
        if not self.check_disk_space():
            return False
        
        zip_config = self.config.get('zip_file', {})
        if not zip_config:
            st.error("❌ ZIP configuration not found")
            return False
        
        zip_path = self.models_dir / zip_config.get('filename', 'models.zip')
        
        # Download zip file
        if not zip_path.exists():
            st.info("📦 Downloading model archive...")
            if not self.download_file(zip_config.get('id'), str(zip_path)):
                return False
        
        # Extract zip
        try:
            with st.spinner("📂 Extracting model files..."):
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(zip_config.get('extract_to', 'models'))
                st.success("✅ Models extracted successfully!")
                
                # Hapus file zip setelah extract
                zip_path.unlink()
                return True
                
        except Exception as e:
            st.error(f"❌ Error extracting models: {str(e)}")
            return False
    
    def check_missing_files(self):
        """Cek file model mana yang belum ada"""
        required_files = self.config.get('required_files', [])
        existing_files = []
        missing_files = []
        
        for filename in required_files:
            if (self.models_dir / filename).exists():
                existing_files.append(filename)
            else:
                missing_files.append(filename)
        
        return existing_files, missing_files
    
    def get_model_size_info(self):
        """Dapatkan informasi ukuran model"""
        total_size = 0
        file_sizes = {}
        
        for filename in self.config.get('required_files', []):
            filepath = self.models_dir / filename
            if filepath.exists():
                size = filepath.stat().st_size
                file_sizes[filename] = size
                total_size += size
        
        return file_sizes, total_size
    
    def show_download_ui(self):
        """Tampilkan UI untuk download"""
        
        existing_files, missing_files = self.check_missing_files()
        
        # Tampilkan status
        if existing_files:
            st.success(f"✅ {len(existing_files)} models ready")
        
        if missing_files:
            st.warning(f"⚠️ {len(missing_files)} models need to be downloaded")
            
            # Tampilkan estimasi ukuran
            total_size_mb = self.config.get('total_size_mb', 200)
            st.info(f"📊 Estimated download size: ~{total_size_mb}MB (may take a few minutes)")
            
            # Pilihan metode download
            download_method = st.radio(
                "Choose download method:",
                ["Download all files (Recommended)", "Download as ZIP (Faster)"],
                key="download_method"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⬇️ Start Download", type="primary", use_container_width=True):
                    return download_method
            with col2:
                if st.button("🔄 Retry", use_container_width=True):
                    st.rerun()
        
        return None


def check_and_download_models():
    """Fungsi utama untuk mengecek dan mendownload models"""
    
    downloader = ModelDownloader()
    
    # Cek file yang ada dan yang missing
    existing_files, missing_files = downloader.check_missing_files()
    
    # Tampilkan status di sidebar
    with st.sidebar:
        with st.expander("📦 Model Status", expanded=False):
            if existing_files:
                st.success(f"✅ {len(existing_files)} models ready")
                for f in existing_files[:5]:  # Tampilkan max 5
                    st.caption(f"  • {f}")
                if len(existing_files) > 5:
                    st.caption(f"  ... and {len(existing_files) - 5} more")
            
            if missing_files:
                st.warning(f"⚠️ {len(missing_files)} models missing")
                for f in missing_files[:5]:
                    st.caption(f"  • {f}")
                if len(missing_files) > 5:
                    st.caption(f"  ... and {len(missing_files) - 5} more")
    
    # Jika ada file yang missing, tampilkan UI download
    if missing_files:
        st.warning("⚠️ Model files not found. Please download them first.")
        
        download_method = downloader.show_download_ui()
        
        if download_method:
            if "ZIP" in download_method:
                success = downloader.download_zip_and_extract()
            else:
                success = downloader.download_all_files()
            
            if success:
                st.success("✅ All models downloaded successfully!")
                time.sleep(2)
                st.rerun()
            else:
                st.error("❌ Download failed. Please try again.")
                return False
        else:
            st.stop()
    
    return True
