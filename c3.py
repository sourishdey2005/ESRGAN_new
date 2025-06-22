import streamlit as st
import tensorflow as tf
import numpy as np
import kagglehub
from PIL import Image, ImageFilter
import time
import io

st.set_page_config(layout="centered")
st.title("🖼️ Ultra-Resolution Dual-Image Super-Resolution (4K/8K)")
st.markdown("Upload **two** low-res images. Get ESRGAN enhanced + 4K or 8K ultra-res output.")

# ---------------------------------------------
# Utility Functions
# ---------------------------------------------

def preprocess_image(image_bytes):
    image = tf.image.decode_image(image_bytes, channels=3)
    image.set_shape([None, None, 3])
    size = tf.convert_to_tensor(image.shape[:-1]) // 4 * 4
    image = tf.image.resize_with_crop_or_pad(image, size[0], size[1])
    image = tf.cast(image, tf.float32)
    return tf.expand_dims(image, 0)

def fuse_images(img1, img2):
    min_shape = tf.minimum(tf.shape(img1)[1:3], tf.shape(img2)[1:3])
    img1 = tf.image.resize(img1, min_shape)
    img2 = tf.image.resize(img2, min_shape)
    return (img1 + img2) / 2.0

def tensor_to_image(tensor):
    tensor = tf.squeeze(tensor, axis=0)
    tensor = tf.clip_by_value(tensor, 0, 255)
    return Image.fromarray(tf.cast(tensor, tf.uint8).numpy())

def enhance_clarity(pil_image):
    return pil_image.filter(ImageFilter.UnsharpMask(radius=1.5, percent=160, threshold=1))

def upscale_to_resolution(img: Image.Image, target_res: str = "4K") -> Image.Image:
    if target_res == "4K":
        return img.resize((3840, 2160), Image.LANCZOS)
    elif target_res == "8K":
        return img.resize((7680, 4320), Image.LANCZOS)
    else:
        return img

# ---------------------------------------------
# Load ESRGAN Model from KaggleHub
# ---------------------------------------------

@st.cache_resource
def load_model():
    model_path = kagglehub.model_download("kaggle/esrgan-tf2/tensorFlow2/esrgan-tf2")
    return tf.saved_model.load(model_path)

model = load_model()

# ---------------------------------------------
# App Logic
# ---------------------------------------------

uploaded_files = st.file_uploader("📤 Upload 2 low-resolution images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

res_choice = st.selectbox("🎯 Choose Target Resolution", ["4K", "8K"], index=0)

if uploaded_files and len(uploaded_files) == 2:
    img1 = preprocess_image(uploaded_files[0].read())
    img2 = preprocess_image(uploaded_files[1].read())

    fused_input = fuse_images(img1, img2)

    st.image(tensor_to_image(fused_input), caption="🧬 Fused Image", use_container_width=True)

    with st.spinner("🚀 Running ESRGAN..."):
        start = time.time()
        sr_output = model(fused_input)
        end = time.time()

    esrgan_image = tensor_to_image(sr_output)
    upscaled_image = upscale_to_resolution(esrgan_image, res_choice)
    final_output = enhance_clarity(upscaled_image)

    tab1, tab2 = st.tabs(["🔬 ESRGAN Output", "🎯 Final Ultra-HD Output"])

    with tab1:
        st.image(esrgan_image, caption="Initial ESRGAN Output", use_container_width=True)

    with tab2:
        st.image(final_output, caption=f"Final {res_choice} Output", use_container_width=True)

    # Sharpness score (TV)
    sharpness_score = tf.image.total_variation(sr_output).numpy().mean()
    st.markdown(f"📏 **Estimated Sharpness (TV):** `{sharpness_score:.2f}`")

    # Download
    buf = io.BytesIO()
    final_output.save(buf, format="PNG")
    st.download_button(
        "⬇️ Download Ultra-Res Image",
        data=buf.getvalue(),
        file_name=f"super_res_{res_choice.lower()}.png",
        mime="image/png"
    )

    st.success(f"✅ Done! Ultra-resolution image ({res_choice}) ready.")
else:
    st.info("Please upload **exactly two** images to proceed.")
