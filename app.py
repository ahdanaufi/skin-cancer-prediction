# ==============================================================================
# PART 2: STREAMLIT WEB APPLICATION (ONNX RUNTIME VERSION)
# ==============================================================================

import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort

# Konfigurasi Halaman Web
st.set_page_config(page_title="Deteksi Kanker Kulit AI", page_icon="🔬", layout="centered")

st.title("🔬 Deteksi Kanker Kulit & Lesi Dermoskopi")
st.write("Aplikasi prediksi medis berbasis model Deep Learning InceptionV3 (Dataset HAM10000).")

# Memuat Model ONNX
@st.cache_resource
def load_onnx_model():
    # Membuka sesi model ONNX menggunakan CPU Runtime server yang sangat ringan
    session = ort.InferenceSession("best_inceptionv3_ham10000.onnx")
    return session

try:
    ort_session = load_onnx_model()
    input_name = ort_session.get_inputs()[0].name
except Exception as e:
    st.error(f"Gagal memuat file model ONNX. Pastikan 'best_inceptionv3_ham10000.onnx' berada di folder yang sama. Error: {e}")
    st.stop()

# Daftar Kelas & Deskripsi Medis
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

uploaded_file = st.file_uploader("Unggah foto lesi kulit makro/dermoskopi (Format: PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Gambar yang Anda Unggah', use_column_width=True)
    
    st.write("⏳ *Sedang memproses struktur piksel citra via ONNX...*")
    
    # Prapemrosesan Gambar
    IMG_WIDTH, IMG_HEIGHT = 120, 120
    image_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(image_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Input shape: [1, 120, 120, 3]

    # Eksekusi Prediksi menggunakan ONNX Runtime
    onnx_inputs = {input_name: img_array}
    raw_predictions = ort_session.run(None, onnx_inputs)
    predictions = raw_predictions[0][0]  # Mengambil array probabilitas kelas
    
    # Keputusan Hasil
    best_idx = np.argmax(predictions)
    label_final = CLASS_LABELS[best_idx]
    confidence_score = predictions[best_idx] * 100

    # Output UI
    st.success("🎉 Analisis Prediksi Sembuh/Selesai!")
    st.subheader(f"Klasifikasi Terdeteksi: **{label_final}**")
    st.metric(label="Confidence Score", value=f"{confidence_score:.2f}%")
    
    st.info(DESKRIPSI_KELAS.get(label_final, "Informasi medis tidak tersedia."))
    
    st.write("---")
    st.subheader("Distribusi Probabilitas Penyakit:")
    for i, label in enumerate(CLASS_LABELS):
        score = predictions[i] * 100
        st.write(f"**{label}** ({score:.2f}%)")
        st.progress(int(score))

    st.warning("⚠️ **Disclaimer Medis:** Hasil analisis berbasis Kecerdasan Buatan (AI) ini ditujukan untuk riset dan edukasi dengan estimasi akurasi global ~77%. Hasil ini bukan merupakan diagnosis final dokter medis.")
