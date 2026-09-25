import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Helmet Detection",
    page_icon="🪖",
    layout="centered"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🪖 Helmet Detection Using YOLO")
st.write(
    "Upload an image to detect whether two-wheeler riders "
    "are wearing helmets."
)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

MODEL_PATH = "models/best.pt"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found: {MODEL_PATH}")
    st.stop()

model = YOLO(MODEL_PATH)


# --------------------------------------------------
# Confidence threshold
# --------------------------------------------------

confidence = st.slider(
    "Detection Confidence",
    min_value=0.10,
    max_value=1.00,
    value=0.25,
    step=0.05
)


# --------------------------------------------------
# Image uploader
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# Detection
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as temp_file:

        image.convert("RGB").save(temp_file.name)
        temp_image_path = temp_file.name

    # Run YOLO detection
    results = model.predict(
        source=temp_image_path,
        conf=confidence,
        save=False
    )

    result = results[0]

    # Generate annotated image
    annotated_image = result.plot()

    st.subheader("Detection Result")

    st.image(
        annotated_image,
        channels="BGR",
        use_container_width=True
    )

    # --------------------------------------------------
    # Display detected objects
    # --------------------------------------------------

    if len(result.boxes) == 0:

        st.warning("No helmet-related objects detected.")

    else:

        st.subheader("Detected Objects")

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence_score = float(box.conf[0])

            class_name = model.names[class_id]

            st.write(
                f"**{class_name}** — "
                f"Confidence: {confidence_score:.2%}"
            )

    # Delete temporary file
    os.remove(temp_image_path)