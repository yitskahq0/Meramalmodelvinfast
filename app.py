import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import urllib.request
# Load Model
@st.cache_resource
def load_model():
    url = "https://drive.proton.me/urls/7ZGSJVM8TG#jECOM9nLyuI2
    model_path = "vinfast_cnn_model1.keras"

    # Cek apakah file model sudah ada
    if not os.path.exists(model_path):
        st.write("📥 Downloading model, please wait...")
        urllib.request.urlretrieve(url, model_path)

    model = tf.keras.models.load_model(model_path, compile=False)
    return model




model = load_model()
class_names = ['VF3', 'VF5', 'VF6', 'VF7']

# Dapatkan ukuran input model
input_shape = model.input_shape[1:3]

# UI
st.title("Klasifikasi Mobil VinFast ")
st.write("Upload gambar mobil untuk memprediksi modelnya (VF3, VF5, VF6, VF7)")

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    # Preprocessing
    img = image.resize(input_shape)
    img_array = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

    # Prediksi
    with st.spinner("Saya Ramalll duluu..."):
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

    st.success(f"**Prediksi ku:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")










