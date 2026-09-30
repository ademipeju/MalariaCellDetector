import cv2 as cv
import numpy as np
import streamlit as st
import tensorflow as tf

# Streamlit page configuration
st.set_page_config(page_title="Malaria Cell Classifier", layout="centered")

# Cache model loading so it executes only once
@st.cache_resource
def load_tflite_model(model_path: str = "malaria_model.tflite"):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

interpreter, input_details, output_details = load_tflite_model()

# Class mapping (0 = Parasitized, 1 = Uninfected)
CLASS_NAMES = ["Parasitized", "Uninfected"]

def is_valid_blood_smear(image_bgr: np.ndarray) -> bool:
    """
    Multi-parameter optical gatekeeper:
    1. Background luminance & contrast uniformity (brightfield microscopy).
    2. Edge complexity check via Laplacian variance.
    3. Giemsa/Wright stain hue & saturation clustering.
    """
    # 1. Luminance & Uniformity
    gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    mean_lum = float(np.mean(gray))
    std_lum = float(np.std(gray))

    # Brightfield microscopy background bounds
    if mean_lum < 100 or mean_lum > 245:
        return False
    if std_lum > 75:  # Blocks cluttered, textured natural scenes
        return False

    # 2. Edge / Texture Complexity
    laplacian_var = float(cv.Laplacian(gray, cv.CV_64F).var())
    if laplacian_var > 650 or laplacian_var < 5:
        return False

    # 3. HSV Color Stain Analysis
    hsv = cv.cvtColor(image_bgr, cv.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]

    # Giemsa/Wright stain (pink-to-purple hue window, moderate saturation)
    stain_pixels = (h >= 120) & (h <= 170) & (s >= 25) & (s <= 200)
    stain_ratio = float(np.count_nonzero(stain_pixels)) / float(image_bgr.shape[0] * image_bgr.shape[1])

    # Isolated erythrocyte patch stained cellular bounds
    if stain_ratio < 0.12 or stain_ratio > 0.85:
        return False

    return True

# UI Headers
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧬 Malaria Cell Image Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a blood smear image to detect if the cell is infected with malaria or not.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload a cell image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Decode uploaded image buffer
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)

    if image is None:
        st.error("❌ Could not load or decode the image.")
    else:
        # Optical Quality Gatekeeper Check
        if not is_valid_blood_smear(image):
            st.error(
                "⚠️ **Input Rejected:** This image does not match the optical characteristics "
                "of a Giemsa-stained microscopic blood smear (brightfield slide illumination and stained erythrocyte morphology). "
                "Please upload a valid microscope cell image."
            )
        else:
            # Preprocessing
            img = cv.resize(image, (128, 128))
            img = img.astype("float32") / 255.0
            img_array = np.expand_dims(img, axis=0)

            # TFLite Inference
            interpreter.set_tensor(input_details[0]["index"], img_array)
            interpreter.invoke()
            prediction = float(interpreter.get_tensor(output_details[0]["index"])[0][0])

            # Binary threshold evaluation
            if prediction < 0.5:
                pred_class = CLASS_NAMES[0]
                confidence = (1 - prediction) * 100
            else:
                pred_class = CLASS_NAMES[1]
                confidence = prediction * 100

            # Output Cards
            st.success(f"🧪 Prediction: **{pred_class}**")
            st.info(f"🔍 Confidence: **{confidence:.2f}%**")
            st.markdown("<small>Note: Classification performed via lightweight TFLite edge model.</small>", unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <hr style='margin: 20px 0;'>
    <small>
      <b>Developers:</b> O.A. Ogunsola & Colleagues<br>
      <b>Project:</b> Deep learning point-of-care malaria cell diagnostic tool
    </small>
    """,
    unsafe_allow_html=True,
)