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
    Robust 4-point gatekeeper for blood smear microscopy:
    1. Rejects blank/dark images.
    2. Checks Giemsa optical signature (Red > Green across cellular regions).
    3. Laplacian texture variance: Rejects high-complexity natural photos.
    4. Hue & Saturation clustering specific to Giemsa/Wright staining.
    """
    h_img, w_img = image_bgr.shape[:2]
    total_pixels = float(h_img * w_img)

    gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(image_bgr, cv.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    mean_brightness = float(np.mean(v))
    if mean_brightness < 20 or mean_brightness > 248:
        return False

    # Texture complexity check: cuts out cluttered natural photos
    lap_var = float(cv.Laplacian(gray, cv.CV_64F).var())
    if lap_var > 750:
        return False

    # Separate cells from black background masks and white slide fields
    non_bg_mask = (v >= 30) & ~((v > 235) & (s < 20))
    cell_pixel_count = int(np.count_nonzero(non_bg_mask))

    if cell_pixel_count / total_pixels < 0.10:
        return False

    # Giemsa stain signature: Red > Green in RGB, Hue in [118, 175], Saturation >= 25
    b, g, r = image_bgr[:, :, 0], image_bgr[:, :, 1], image_bgr[:, :, 2]
    giemsa_color_rule = (r.astype(int) > (g.astype(int) + 8))

    valid_stain_mask = (h >= 118) & (h <= 175) & (s >= 25) & giemsa_color_rule & non_bg_mask
    stain_ratio = float(np.count_nonzero(valid_stain_mask)) / float(cell_pixel_count)

    if stain_ratio < 0.20:
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
                "of a Giemsa-stained microscopic blood smear. Please upload a valid erythrocyte patch."
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
      <b>Developers:</b> O.A. Ogunsola<br>
      <b>Project:</b> Deep learning point-of-care malaria cell diagnostic tool
    </small>
    """,
    unsafe_allow_html=True,
)