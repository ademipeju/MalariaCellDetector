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
    Validates segmented erythrocyte patches:
    1. Accommodates segmented black background masks.
    2. Ensures a prominent cellular body is detected.
    3. Verifies Giemsa/Wright stain chromaticity within the cell region.
    """
    # Convert BGR to HSV
    hsv = cv.cvtColor(image_bgr, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    # Foreground cell isolation (non-black pixels have v > 35)
    cell_mask = v > 35
    cell_pixel_count = int(np.count_nonzero(cell_mask))
    total_pixels = int(image_bgr.shape[0] * image_bgr.shape[1])

    # Segmented cell must occupy between 15% and 98% of the image patch
    cell_coverage = cell_pixel_count / total_pixels
    if cell_coverage < 0.15 or cell_coverage > 0.98:
        return False

    # Check Giemsa pink-to-purple stain range (H in [115, 178], S >= 20) inside the segmented cell
    stain_mask = (h >= 115) & (h <= 178) & (s >= 20) & cell_mask
    stain_ratio_in_cell = float(np.count_nonzero(stain_mask)) / float(cell_pixel_count)

    # Cellular body must contain at least 25% stained hue composition
    if stain_ratio_in_cell < 0.25:
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
                "of a Giemsa-stained microscopic blood smear patch. Please upload a valid erythrocyte patch."
            )
        else:
            # Preprocessing
            img = cv.resize(image, (128, 128))
            img = img.astype("float32") / 255.0
            img_array = np.expand_dims(img, axis=0)

            # Run TFLite inference
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