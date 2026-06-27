# ==============================================================================
# PART 2: STREAMLIT WEB APPLICATION
# ==============================================================================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import urllib.request

# Konfigurasi Halaman Web
st.set_page_config(page_title="Deteksi Kanker Kulit AI", page_icon="🔬", layout="centered")

st.title("🔬 Deteksi Kanker Kulit & Lesi Dermoskopi")
st.write("Aplikasi prediksi medis berbasis model Deep Learning InceptionV3 (Dataset HAM10000).")

# Memuat Model TF Lite (Cepat, Ringan, Hemat RAM Server Cloud)
@st.cache_resource
def load_model():
    model_path = "best_inceptionv3_ham10000.tflite"
    
    # KUNCI UTAMA: Jika model belum ada di server Streamlit, download otomatis
    if not os.path.exists(model_path):
        with st.spinner("Mengunduh model dari server (hanya dilakukan sekali)..."):
            # GANTI URL DI BAWAH INI dengan Link Address yang kamu copy dari Release tadi!
            url = "sha256:e0e3d6cabcdc3a6bfcea3f655de52dcd56bea61ad5e50d05242d2aebd405f627"
            urllib.request.urlretrieve(url, model_path)
            
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    st.error(f"Gagal memuat file model. Pastikan 'best_inceptionv3_ham10000.tflite' berada di folder yang sama. Error: {e}")
    st.stop()

# Daftar Kelas & Deskripsi Medis Ringkas
CLASS_LABELS = ['Actinic Keratosis', 'Basal Cell Carcinoma', 'Benign Keratosis', 'Dermatofibroma', 'Melanocytic Nevi', 'Melanoma', 'Vascular Lesions']
DESKRIPSI_KELAS = {
    'Melanoma': "⚠️ **Ganas (Malignant).** Jenis kanker kulit paling agresif. Diperlukan penanganan dan biopsi medis segera.",
    'Melanocytic Nevi': "🟢 **Jinak (Benign).** Tahi lalat biasa yang terbentuk dari melanosit. Umumnya aman dan tidak berbahaya.",
    'Basal Cell Carcinoma': "⚠️ **Ganas (Malignant).** Kanker kulit non-melanoma yang tumbuh lambat namun merusak jaringan lokal jika diabaikan.",
    'Benign Keratosis': "🟢 **Jinak (Benign).** Tumor jinak superfisial kulit yang biasa muncul karena faktor penuaan.",
    'Actinic Keratosis': "🟡 **Pre-Kanker.** Lesi kasar bersisik yang berpotensi berubah menjadi keganasan sel skuamosa di masa depan.",
    'Vascular Lesions': "🟢 **Jinak (Benign).** Tumor atau pertumbuhan pembuluh darah jinak seperti angioma.",
    'Dermatofibroma Mini': "🟢 **Jinak (Benign).** Nodul kulit jinak kecil yang biasanya muncul di area kaki."
}

# Interface Upload Gambar oleh Pengguna
uploaded_file = st.file_uploader("Unggah foto lesi kulit makro/dermoskopi (Format: PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Gambar yang Anda Unggah', use_column_width=True)
    
    st.write("⏳ *Sedang memproses struktur piksel citra...*")
    
    # Prapemrosesan Gambar agar Identik dengan Spesifikasi Pipeline Latih
    IMG_WIDTH, IMG_HEIGHT = 120, 120
    image_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(image_resized, dtype=np.float32) / 255.0  # Normalisasi standardisasi 1./255
    img_array = np.expand_dims(img_array, axis=0)                  # Ekspansi dimensi batch

    # Eksekusi Prediksi Model (Inference)
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Pengambilan Keputusan Kelas Tertinggi
    best_idx = np.argmax(predictions)
    label_final = CLASS_LABELS[best_idx]
    confidence_score = predictions[best_idx] * 100

    # Menampilkan Hasil Akhir ke Pengguna
    st.success("🎉 Analisis Prediksi Sembuh/Selesai!")
    st.subheader(f"Klasifikasi Terdeteksi: **{label_final}**")
    st.metric(label="Confidence Score", value=f"{confidence_score:.2f}%")
    
    # Tampilkan Edukasi Medis Terkait Kelas
    st.info(DESKRIPSI_KELAS.get(label_final, "Informasi medis tidak tersedia."))
    
    # Visualisasi Bar Chart Seluruh Probabilitas Kelas
    st.write("---")
    st.subheader("Distribusi Probabilitas Penyakit:")
    for i, label in enumerate(CLASS_LABELS):
        score = predictions[i] * 100
        st.write(f"**{label}** ({score:.2f}%)")
        st.progress(int(score))

    # Regulasi Keamanan Medis (Disclaimer)
    st.warning("⚠️ **Disclaimer Medis:** Hasil analisis berbasis Kecerdasan Buatan (AI) ini ditujukan untuk riset dan edukasi dengan estimasi akurasi global ~90%. Hasil ini bukan merupakan diagnosis final dokter medis.")