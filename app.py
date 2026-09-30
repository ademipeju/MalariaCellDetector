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

       st.markdown(
    """
    <hr style='margin: 20px 0;'>
    <small>
      <b>Developed by:</b> Department of Biochemistry Research Team<br>
      <b>Contributors:</b> O.A. Ogunsola & Colleagues<br>
      <i>Acknowledgments: Initial prototype conceived through training with Women in AI Nigeria.</i>
    </small>
    """,
    unsafe_allow_html=True,
)