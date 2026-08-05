import streamlit as st
from PIL import Image

from src.data_loader import load_image
from src.predict import predict_image

st.set_page_config(
    page_title="Breast Tumor Detection",
    page_icon="🩺",
    layout="centered",
)

st.title("🩺 Breast Tumor Detection")

st.write(
    """
Upload a breast ultrasound image to classify it as **Normal** or **Malignant**
using a Convolutional Neural Network (CNN).
"""
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True,
    )

    if st.button("Predict"):

        with st.spinner("Analyzing image..."):

            img = load_image(uploaded_file)

            label, confidence = predict_image(img)

        st.markdown("---")

        st.subheader("Prediction")

        if label == "Normal":
            st.success(f"🟢 {label}")
        else:
            st.error(f"🔴 {label}")

        st.metric(
            "Confidence",
            f"{confidence*100:.2f} %",
        )

st.markdown("---")

st.markdown(
"""
### Model Information

- TensorFlow / Keras
- Custom CNN
- Input Size: **224 × 224**
- Classes:
    - Normal
    - Malignant
"""
)