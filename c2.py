import streamlit as st
import tensorflow as tf
import numpy as np
import kagglehub
from PIL import Image, ImageFilter
import time
import io

st.set_page_config(layout="centered")
st.title("🔥 Triple-Pass Dual-Image Super Resolution")
st.markdown("Upload **two** low-resolution images. ESRGAN will enhance them across three stages for exceptional clarity.")

# ---------------------------------------------
# Utility functions
# ---------------------------------------------

def preprocess_image(image_bytes):
    image = tf.image.decode_image(image_bytes, channels=3)
    size = (tf.convert_to_tensor(image.shape[:-1]) // 4) * 4
    image = tf.image.crop_to_bounding_box(image, 0, 0, size[0], size[1])
    image = tf.cast(image, tf.float32)
    return tf.expand_dims(image, 0)

def fuse_images(img1, img2):
    min_shape = tf.minimum(tf.shape(img1)[1:3], tf.shape(img2)[1:3])
    img1_resized = tf.image.resize(img1, min_shape)
    img2_resized = tf.image.resize(img2, min_shape)
    fused = (img1_resized + img2_resized) / 2.0
    return tf.cast(fused, tf.float32)

def tensor_to_image(tensor):
    tensor = tf.squeeze(tf.clip_by_value(tensor, 0, 255))
    return Image.fromarray(tf.cast(tensor, tf.uint8).numpy())

def enhance_clarity(pil_image):
    return pil_image.filter(ImageFilter.UnsharpMask(radius=2.0, percent=180, threshold=2))

# ---------------------------------------------
# Load ESRGAN model from KaggleHub
# ---------------------------------------------

@st.cache_resource
def load_model():
    model_path = kagglehub.model_download("kaggle/esrgan-tf2/tensorFlow2/esrgan-tf2")
    return tf.saved_model.load(model_path)

model = load_model()

# ---------------------------------------------
# Main App Logic
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
    st.image(tensor_to_image(fused_input), caption="🧬 Fused Input", use_column_width=True)

    with st.spinner("🔧 Enhancing resolution in 3 ESRGAN stages..."):
        start = time.time()
        sr1 = model(fused_input)
        sr2 = model(sr1)
        sr3 = model(sr2)
        duration = time.time() - start

    final_output = tensor_to_image(sr3)
    enhanced_output = enhance_clarity(final_output)

    tabs = st.tabs(["Final ESRGAN Output", "Enhanced Clarity Output"])

    with tabs[0]:
        st.image(final_output, caption=f"🖼️ After 3x ESRGAN passes (Time: {duration:.2f}s)", use_column_width=True)

    with tabs[1]:
        st.image(enhanced_output, caption="✨ Final Output + Clarity Boost", use_column_width=True)

    sharpness_score = tf.image.total_variation(sr3).numpy().mean()
    st.markdown(f"🧠 **Blind Sharpness Estimate (TV):** `{sharpness_score:.2f}`")

    # Create download button
    buf = io.BytesIO()
    enhanced_output.save(buf, format="PNG")
    byte_data = buf.getvalue()

    st.download_button(
        label="⬇️ Download Final Enhanced Image",
        data=byte_data,
        file_name="final_enhanced_sr.png",
        mime="image/png"
    )

    st.success("✅ Done! Triple-pass enhancement complete.")
else:
    st.info("Please upload exactly two low-resolution images to proceed.")
