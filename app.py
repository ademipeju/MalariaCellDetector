import cv2 as cv
import numpy as np
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="Malaria Cell Classifier", layout="centered")


# Load the fine-tuned 3-class TFLite model
@st.cache_resource
def load_tflite_model(model_path: str = "malaria_model_v2.tflite"):
  interpreter = tf.lite.Interpreter(model_path=model_path)
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  return interpreter, input_details, output_details


interpreter, input_details, output_details = load_tflite_model()

# 3-Class definitions matching the notebook labels
CLASS_NAMES = ["Parasitized", "Uninfected", "Non-Smear / Invalid Image"]

st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>🧬 Malaria Cell Image"
    " Classifier</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center;'>Upload an image to screen for malaria"
    " infection status.</p>",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "📤 Upload an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

  file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
  image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)

  if image is None:
    st.error("❌ Could not decode the uploaded image file.")
  else:
    # Preprocessing: resize to 128x128 and scale pixel values to [0, 1]
    img = cv.resize(image, (128, 128))
    img = img.astype("float32") / 255.0
    img_array = np.expand_dims(img, axis=0)

    # Execute TFLite inference
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()

    # Extract 3-class Softmax probabilities
    probabilities = interpreter.get_tensor(output_details[0]["index"])[0]
    predicted_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_idx]) * 100

    # Route output based on classification
    if predicted_idx == 2:
      st.error(
          f"⚠️ **Input Rejected:** The model identified this image as non-smear"
          f" / out-of-distribution (Confidence: {confidence:.2f}%). Please"
          " upload a valid microscopic blood smear."
      )
    elif predicted_idx == 0:
      st.error("🧪 Prediction: **Parasitized**")
      st.info(f"🔍 Confidence: **{confidence:.2f}%**")
    else:
      st.success("🧪 Prediction: **Uninfected**")
      st.info(f"🔍 Confidence: **{confidence:.2f}%**")

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