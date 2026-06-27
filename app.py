import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import os
import urllib.request

# Konfigurasi Halaman Web
st.set_page_config(page_title="Deteksi Kanker Kulit AI", page_icon="🔬", layout="centered")

st.title("🔬 Deteksi Kanker Kulit & Lesi Dermoskopi")
st.write("Aplikasi prediksi medis berbasis model Deep Learning InceptionV3 (Dataset HAM10000).")

@st.cache_resource
def load_onnx_model():
    model_path = "best_inceptionv3_ham10000.onnx"
    
    # Fungsi auto-download dari Release GitHub jika file belum ada di server
    if not os.path.exists(model_path):
        with st.spinner("Mengunduh model ONNX dari server (Proses ini hanya dilakukan sekali saat web pertama kali dibuka)..."):
            # ⚠️ PASTIKAN URL DI BAWAH INI SUDAH BENAR (Sesuai hasil copy-right click dari Assets Release)
            url = "https://github.com/ahdanaufi/skin-cancer-prediction/releases/download/v1.0.0/best_inceptionv3_ham10000.onnx"
            urllib.request.urlretrieve(url, model_path)
            
    session = ort.InferenceSession(model_path)
    return session

try:
    ort_session = load_onnx_model()
    input_name = ort_session.get_inputs()[0].name
except Exception as e:
    st.error(f"Gagal memuat file model ONNX dari server. Error: {e}")
    st.stop()

CLASS_LABELS = ['Actinic Keratosis', 'Basal Cell Carcinoma', 'Benign Keratosis', 'Dermatofibroma', 'Melanocytic Nevi', 'Melanoma', 'Vascular Lesions']

DESKRIPSI_KELAS = {
    'Melanoma': "⚠️ **Ganas (Malignant).** Jenis kanker kulit paling agresif. Diperlukan penanganan dan biopsi medis segera.",
    'Melanocytic Nevi': "🟢 **Jinak (Benign).** Tahi lalat biasa yang terbentuk dari melanosit. Umumnya aman dan tidak berbahaya.",
    'Basal Cell Carcinoma': "⚠️ **Ganas (Malignant).** Kanker kulit non-melanoma yang tumbuh lambat namun merusak jaringan lokal jika diabaikan.",
    'Benign Keratosis': "🟢 **Jinak (Benign).** Tumor jinak superfisial kulit yang biasa muncul karena faktor penuaan.",
    'Actinic Keratosis': "🟡 **Pre-Kanker.** Lesi kasar bersisik yang berpotensi berubah menjadi keganasan sel skuamosa di masa depan.",
    'Vascular Lesions': "🟢 **Jinak (Benign).** Tumor atau pertumbuhan pembuluh darah jinak seperti angioma.",
    'Dermatofibroma': "🟢 **Jinak (Benign).** Nodul kulit jinak kecil yang biasanya muncul di area kaki."
}

# ==============================================================================
# FITUR BARU: PILIHAN INPUT (KAMERA ATAU UPLOAD)
# ==============================================================================
st.write("---")
menu_input = st.radio("Pilih Metode Input Gambar:", ("📷 Ambil Foto via Kamera", "📁 Upload File Gambar"))

source_gambar = None

if menu_input == "📷 Ambil Foto via Kamera":
    # Mengaktifkan kamera internal HP / Webcam Laptop
    source_gambar = st.camera_input("Posisikan lesi kulit/tahi lalat tepat di tengah kamera dan ambil gambar")
else:
    source_gambar = st.file_uploader("Unggah foto lesi kulit (Format: PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

# ==============================================================================
# PROSES PREDIKSI (Jalan jika ada gambar yang masuk dari Kamera / Upload)
# ==============================================================================
if source_gambar is not None:
    image = Image.open(source_gambar).convert('RGB')
    
    # Jika dari file uploader kita tampilkan gambarnya, 
    # kalau dari st.camera_input tidak perlu karena kameranya sudah menampilkan gambarnya otomatis.
    if menu_input == "📁 Upload File Gambar":
        st.image(image, caption='Gambar yang Anda Unggah', use_column_width=True)
    
    st.write("⏳ *Sedang memproses gambar dan menganalisis struktur piksel...*")
    
    # Prapemrosesan Gambar
    IMG_WIDTH, IMG_HEIGHT = 120, 120
    image_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(image_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Jalankan Model ONNX
    onnx_inputs = {input_name: img_array}
    raw_predictions = ort_session.run(None, onnx_inputs)
    predictions = raw_predictions[0][0]
    
    best_idx = np.argmax(predictions)
    label_final = CLASS_LABELS[best_idx]
    confidence_score = predictions[best_idx] * 100

    # Menampilkan Output ke Halaman Web
    st.success("🎉 Analisis Prediksi Selesai!")
    st.subheader(f"Klasifikasi Terdeteksi: **{label_final}**")
    st.metric(label="Confidence Score", value=f"{confidence_score:.2f}%")
    
    st.info(DESKRIPSI_KELAS.get(label_final, "Informasi medis tidak tersedia."))
    
    st.write("---")
    st.subheader("Distribusi Probabilitas Penyakit:")
    for i, label in enumerate(CLASS_LABELS):
        score = predictions[i] * 100
        st.write(f"**{label}** ({score:.2f}%)")
        st.progress(int(score))

    st.warning("⚠️ **Disclaimer Medis:** Hasil analisis berbasis Kecerdasan Buatan (AI) ini ditujukan untuk riset dengan estimasi akurasi global ~77%.")
