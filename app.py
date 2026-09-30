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
    Quality gatekeeper: Checks whether the image matches expected 
    microscopic brightfield illumination and Giemsa/Wright stain hues.
    """
    # Convert BGR to HSV color space
    hsv = cv.cvtColor(image_bgr, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    # 1. Background Brightness Check (smears have high average illumination)
    mean_val = np.mean(v)
    if mean_val < 70 or mean_val > 250:
        return False

    # 2. Stain Color Range: Pink/Purple hues characteristic of Giemsa stain
    # In OpenCV HSV: H is 0-179. Typical erythrocyte/Giemsa stain spans ~120 to 175
    stain_mask = cv.inRange(hsv, np.array([115, 20, 40]), np.array([178, 255, 255]))
    stain_ratio = np.count_nonzero(stain_mask) / stain_mask.size

    # A valid cell patch typically contains between 5% and 85% stained cellular material
    if stain_ratio < 0.05 or stain_ratio > 0.85:
        return False

    return True

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
        # Quality Gatekeeper Check
        if not is_valid_blood_smear(image):
            st.error("⚠️️ **Invalid Input:** This image does not match the optical characteristics of a Giemsa-stained microscopic blood smear. Please upload a valid erythrocyte patch.")
        else:
            # Preprocessing
            img = cv.resize(image, (128, 128))
            img = img.astype("float32") / 255.0
            img_array = np.expand_dims(img, axis=0)

            # Run inference
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

            # Display results
            st.success(f"🧪 Prediction: **{pred_class}**")
            st.info(f"🔍 Confidence: **{confidence:.2f}%**")
            st.markdown("<small>Note: This tool uses a lightweight TFLite model trained on malaria cell images.</small>", unsafe_allow_html=True)

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