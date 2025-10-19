import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os


# Load Model

@st.cache_resource
def load_model():
    model_path = "vinfast_cnn_model1.keras"

    # Kalau model belum ada di local, download dari Google Drive
    if not os.path.exists(model_path):
        st.info("Sabar yaa... lagi unduh model dari Google Drive nihh bentarr...")
        file_id = "1LuCr_vda0oOz_7-55REALzTi2gexq5U9" 
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)
        
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
