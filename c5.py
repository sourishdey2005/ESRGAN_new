import streamlit as st
import tensorflow as tf
import numpy as np
import kagglehub
from PIL import Image, ImageFilter
import time
import io
import gc
import os

# ----------------------------
# TF Config for Memory Safety
# ----------------------------
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

st.set_page_config(layout="centered")
st.title(" ESRGAN Ultra-HD ")
st.markdown("Upload **two low-res images** → ESRGAN → Final 4K/8K sharp image")

# ----------------------------
# Utility Functions
# ----------------------------

def preprocess_image(image_bytes, max_dim=512):
    image = tf.image.decode_image(image_bytes, channels=3)
    image.set_shape([None, None, 3])
    image = tf.cast(image, tf.float32)

    shape = tf.shape(image)[:2]
    scale = tf.minimum(max_dim / tf.cast(shape[0], tf.float32),
                       max_dim / tf.cast(shape[1], tf.float32))
    new_size = tf.cast(tf.cast(shape, tf.float32) * scale, tf.int32)
    image = tf.image.resize(image, new_size)

    # Ensure dimensions divisible by 4
    h, w = new_size[0] - new_size[0] % 4, new_size[1] - new_size[1] % 4
    image = tf.image.resize_with_crop_or_pad(image, h, w)
    return tf.expand_dims(image, 0)

def fuse_images(img1, img2):
    shape = tf.minimum(tf.shape(img1)[1:3], tf.shape(img2)[1:3])
    img1_resized = tf.image.resize(img1, shape)
    img2_resized = tf.image.resize(img2, shape)
    return (img1_resized + img2_resized) / 2.0

def tensor_to_pil(tensor):
    tensor = tf.squeeze(tf.clip_by_value(tensor, 0, 255), axis=0)
    return Image.fromarray(tf.cast(tensor, tf.uint8).numpy())

def upscale_to_resolution(img: Image.Image, resolution: str = "4K") -> Image.Image:
    target_size = (3840, 2160) if resolution == "4K" else (7680, 4320)
    return img.resize(target_size, Image.LANCZOS)

def sharpen_image(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=1))

# ----------------------------
# Load ESRGAN Model
# ----------------------------

@st.cache_resource
def load_model():
    path = kagglehub.model_download("kaggle/esrgan-tf2/tensorFlow2/esrgan-tf2")
    loaded = tf.saved_model.load(path)

    @tf.function
    def super_res(tensor):
        return loaded(tensor)

    return super_res

model = load_model()

# ----------------------------
# UI Input
# ----------------------------

uploaded_files = st.file_uploader(" Upload exactly two low-res images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
res_choice = st.selectbox(" Final Output Resolution", ["4K", "8K"], index=0)

# ----------------------------
# Main Logic
# ----------------------------

if uploaded_files and len(uploaded_files) == 2:
    img1 = preprocess_image(uploaded_files[0].read(), max_dim=512)
    img2 = preprocess_image(uploaded_files[1].read(), max_dim=512)
    fused = fuse_images(img1, img2)

    st.image(tensor_to_pil(fused), caption=" Fused Input Image", use_container_width=True)

    with st.spinner("🚀 Running ESRGAN enhancement x4..."):
        t0 = time.time()
        enhanced = fused
        for i in range(4):
            enhanced = model(enhanced)
        t1 = time.time()

    base_output = tensor_to_pil(enhanced)
    del img1, img2, fused
    gc.collect()

    with st.spinner(" Upscaling to final resolution and sharpening..."):
        upscale_img = upscale_to_resolution(base_output, res_choice)
        final_img = sharpen_image(upscale_img)

    tabs = st.tabs([" ESRGAN Final Output (x4)", f" {res_choice} + Sharpened"])

    with tabs[0]:
        st.image(base_output, caption="Intermediate Output (After 4 Passes)", use_container_width=True)

    with tabs[1]:
        st.image(final_img, caption=f"{res_choice} Final Result", use_container_width=True)

    sharpness = tf.image.total_variation(
        tf.convert_to_tensor(np.expand_dims(np.array(base_output), 0), dtype=tf.float32)
    ).numpy().mean()
    st.markdown(f" **Sharpness Estimate (TV):** `{sharpness:.2f}`")
    st.markdown(f"⏱ **Time Taken:** `{t1 - t0:.2f}` seconds")

    buf = io.BytesIO()
    final_img.save(buf, format="PNG")
    st.download_button("⬇ Download Final Enhanced Image", buf.getvalue(),
                       file_name=f"ultra_hd_{res_choice.lower()}.png", mime="image/png")

    st.success(" Ultra-HD image generation complete!")

else:
    st.info("Please upload exactly **two images** to continue.")
