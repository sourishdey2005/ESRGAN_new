import streamlit as st
import tensorflow as tf
import numpy as np
import kagglehub
from PIL import Image
import time

st.set_page_config(layout="centered")
st.title("Dual-Image Super Resolution App by CodeKarma")
st.markdown("Upload **two** low-resolution images to generate a high-resolution output .")

# ---------------------------------------------
# Utility functions
# ---------------------------------------------

def preprocess_image(image_bytes):
    image = tf.image.decode_image(image_bytes, channels=3)
    size = (tf.convert_to_tensor(image.shape[:-1]) // 4) * 4
    image = tf.image.crop_to_bounding_box(image, 0, 0, size[0], size[1])
    image = tf.cast(image, tf.float32)
    return tf.expand_dims(image, 0)  # Shape: (1, H, W, 3)

def fuse_images(img1, img2):
    min_shape = tf.minimum(tf.shape(img1)[1:3], tf.shape(img2)[1:3])
    img1_resized = tf.image.resize(img1, min_shape)
    img2_resized = tf.image.resize(img2, min_shape)
    fused = (img1_resized + img2_resized) / 2.0
    return tf.cast(fused, tf.float32)  # Shape preserved: (1, H, W, 3)

def tensor_to_image(tensor):
    tensor = tf.squeeze(tf.clip_by_value(tensor, 0, 255))
    return Image.fromarray(tf.cast(tensor, tf.uint8).numpy())

# ---------------------------------------------
# Load ESRGAN model
# ---------------------------------------------

@st.cache_resource
def load_model():
    model_path = kagglehub.model_download("kaggle/esrgan-tf2/tensorFlow2/esrgan-tf2")
    return tf.saved_model.load(model_path)

model = load_model()

# ---------------------------------------------
# Main Interface
# ---------------------------------------------

uploaded_files = st.file_uploader(
    "📤 Upload two low-resolution images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files and len(uploaded_files) == 2:
    img_bytes_1 = uploaded_files[0].read()
    img_bytes_2 = uploaded_files[1].read()
    img1 = preprocess_image(img_bytes_1)
    img2 = preprocess_image(img_bytes_2)

    fused_input = fuse_images(img1, img2)
    st.image(tensor_to_image(fused_input), caption="🧬 Fused Low-Resolution Input", use_container_width=True)

    with st.spinner("🔧 Upscaling in progress..."):
        start = time.time()
        sr_output = model(fused_input)
        duration = time.time() - start

    st.image(tensor_to_image(sr_output), caption=f"🖼️ Super-Resolved Output (Time: {duration:.2f}s)", use_container_width=True)

    avg_pixel_variance = tf.image.total_variation(sr_output).numpy().mean()
    st.markdown(f"🧠 **Blind Sharpness Estimate (Total Variation):** `{avg_pixel_variance:.2f}`")

    st.success("✅ Done! High-resolution output generated successfully.")
else:
    st.info("Please upload exactly two low-resolution images to proceed.")
