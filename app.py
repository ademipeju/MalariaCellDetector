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
    Broad Domain Gatekeeper:
    Accepts:
      - Single-cell segmented patches (black masked borders, NIH format)
      - Full microscopic slides (circular field-of-view, brightfield background)
    Rejects:
      - General everyday photos (cars, people, scenery, animals, outdoor scenes)
    """
    # 1. Reject solid blank or completely corrupted images
    gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    if np.std(gray) < 8.0:
        return False

    # 2. Reject natural scene texture clutter (Laplacian edge variance)
    # Natural photos (hair, fabric, tree foliage, street clutter) have sharp variance across depth planes.
    lap_var = float(cv.Laplacian(gray, cv.CV_64F).var())
    if lap_var > 1400:
        return False

    # 3. Analyze Color Distribution in HSV and RGB
    hsv = cv.cvtColor(image_bgr, cv.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    b = image_bgr[:, :, 0].astype(int)
    g = image_bgr[:, :, 1].astype(int)
    r = image_bgr[:, :, 2].astype(int)

    # Exclude dark background masks (v < 25) and blown-out pure white backgrounds (v > 245 with s < 15)
    foreground_mask = (v >= 25) & ~((v > 245) & (s < 15))
    fg_pixel_count = int(np.count_nonzero(foreground_mask))

    total_pixels = int(image_bgr.shape[0] * image_bgr.shape[1])
    if (fg_pixel_count / total_pixels) < 0.05:
        return False

    # 4. Giemsa Stain vs Nature Color Check
    # Blood smears: Red channel exceeds Green (erythrocyte pink) OR Blue exceeds Green (purple nuclei/parasites)
    smear_color_rule = ((r > g) | (b > g)) & foreground_mask

    # Hue spectrum: Eosin pinks wrap around 0-25 and 150-179; Giemsa purples sit in 120-155
    stain_hue_rule = ((h <= 25) | (h >= 120)) & (s >= 12) & foreground_mask

    # Reject outdoor scenes with high green foliage/vegetation dominance
    green_dominant_pixels = (g > (r + 10)) & (g > (b + 10)) & foreground_mask
    green_ratio = np.count_nonzero(green_dominant_pixels) / float(fg_pixel_count)
    if green_ratio > 0.12:  # Rejects grass, trees, plants, outdoor green scenes
        return False

    # Check the fraction of valid smear chromaticity
    valid_smear_pixels = smear_color_rule & stain_hue_rule
    stain_fraction = np.count_nonzero(valid_smear_pixels) / float(fg_pixel_count)

    # In both single cells and full slides, at least 25% of the foreground follows smear chromaticity
    if stain_fraction < 0.25:
        return False

    return True

# UI Headers
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧬 Malaria Cell Image Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a blood smear image to detect if the cell is infected with malaria or not.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload a cell or slide image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)

    if image is None:
        st.error("❌ Could not load or decode the image.")
    else:
        # Domain Gatekeeper Check
        if not is_valid_blood_smear(image):
            st.error(
                "⚠️ **Input Rejected:** This image does not appear to be a microscopic blood smear slide. "
                "Please upload a valid Giemsa-stained blood smear or erythrocyte patch."
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
            st.markdown("<small>Note: Inference executed via lightweight TFLite model.</small>", unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <hr style='margin: 20px 0;'>
    <small>
      <b>Developer:</b> O.A. Ogunsola<br>
      <b>Project:</b> Deep learning point-of-care malaria diagnostic tool
    </small>
    """,
    unsafe_allow_html=True,
)