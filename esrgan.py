import streamlit as st
import tensorflow as tf
import numpy as np
import kagglehub
from PIL import Image
import io
import time
import os

st.set_page_config(layout="centered")
st.title("ESRGAN Super Resolution App (KaggleHub)")
st.markdown("Upload an image to upscale it using ESRGAN from KaggleHub.")

# Load model only once
@st.cache_resource
def load_model():
    model_path = kagglehub.model_download("kaggle/esrgan-tf2/tensorFlow2/esrgan-tf2")
    print("Path to model files:", model_path)
    return tf.saved_model.load(model_path)

model = load_model()

def preprocess_image(image_bytes):
    image = tf.image.decode_image(image_bytes, channels=3)
    size = (tf.convert_to_tensor(image.shape[:-1]) // 4) * 4
    image = tf.image.crop_to_bounding_box(image, 0, 0, size[0], size[1])
    image = tf.cast(image, tf.float32)
    return tf.expand_dims(image, 0)

def downscale_image(image):
    size = [image.shape[1] // 4, image.shape[0] // 4]
    image = tf.clip_by_value(image, 0, 255)
    image_pil = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image_pil = image_pil.resize(size, Image.BICUBIC)
    return tf.expand_dims(tf.cast(np.array(image_pil), tf.float32), 0)

def tensor_to_image(tensor):
    tensor = tf.squeeze(tf.clip_by_value(tensor, 0, 255))
    return Image.fromarray(tf.cast(tensor, tf.uint8).numpy())

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image_bytes = uploaded_file.read()
    hr_image = preprocess_image(image_bytes)
    st.image(tensor_to_image(hr_image), caption="Original Image", use_column_width=True)

    with st.spinner("Upscaling..."):
        start = time.time()
        sr_image = model(hr_image)
        duration = time.time() - start

    st.image(tensor_to_image(sr_image), caption=f"Super Resolution (Time: {duration:.2f}s)", use_column_width=True)

    # Downscale test
    lr_image = downscale_image(tf.squeeze(hr_image))
    sr_from_lr = model(lr_image)
    psnr = tf.image.psnr(tf.clip_by_value(sr_from_lr, 0, 255), tf.clip_by_value(hr_image, 0, 255), max_val=255)

    st.markdown(f"🔎 **PSNR (LR→SR vs HR)**: {psnr.numpy()[0]:.2f} dB")
    st.image(tensor_to_image(sr_from_lr), caption="Super Resolution from Downscaled LR", use_column_width=True)

    st.success("Done! You can compare outputs visually above.")
