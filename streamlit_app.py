
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = Path(__file__).resolve().parent / "image_identifier_model.h5"
IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🔍",
    layout="centered",
)

st.title("🔍 AI vs Real Image Detector")
st.write(
    "Upload an image to classify it as a real photograph "
    "or an AI-generated image."
)

@st.cache_resource
def load_detector():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None,
    )
    base.trainable = False

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(224, 224, 3)),
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.load_weights(str(MODEL_PATH))
    return model


uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Selected image", use_container_width=True)

        if st.button("Classify Image", type="primary"):
            with st.spinner("Analyzing image..."):
                model = load_detector()

                resized = image.resize(IMG_SIZE)
                arr = np.asarray(resized, dtype=np.float32) / 255.0
                pred = float(
                    model.predict(arr[np.newaxis, ...], verbose=0)[0][0]
                )

                if pred > 0.5:
                    label = "Real Image"
                    confidence = pred * 100
                else:
                    label = "AI-Generated Image"
                    confidence = (1 - pred) * 100

            st.subheader(f"Result: {label}")
            st.metric("Estimated class confidence", f"{confidence:.2f}%")
            st.caption(
                "This is a model estimate, not a guarantee. "
                "Results may be incorrect for some images."
            )

    except Exception as exc:
        st.error(f"Could not process this image: {exc}")

st.divider()
st.caption("Powered by MobileNetV2 • AI Image Detection Project")
