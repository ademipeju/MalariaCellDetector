# import streamlit as st
# import numpy as np
# import cv2 as cv
# from tensorflow.keras.models import load_model
# from PIL import Image

# # Load model
# model = load_model("malaria_cnn_model.keras")

# # Class labels
# class_names = ['Parasitized', 'Uninfected']  # 0 = Parasitized, 1 = Uninfected

# # Streamlit UI
# st.set_page_config(page_title="Malaria Cell Classifier", layout="centered")
# st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧬 Malaria Cell Image Classifier</h1>", unsafe_allow_html=True)
# st.markdown("<p style='text-align: center;'>Upload a blood smear image to detect if the cell is infected with malaria or not.</p>", unsafe_allow_html=True)

# uploaded_file = st.file_uploader("📤 Upload a cell image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

#     # Decode image
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)

#     if image is None:
#         st.error("Error loading image.")
#     else:
#         # Preprocess
#         img = cv.resize(image, (128, 128))
#         img = img.astype("float32") / 255.0
#         img_array = np.expand_dims(img, axis=0)

#         # Predict
#         prob = float(model.predict(img_array)[0][0])

#         if prob < 0.5:
#             pred_class = "Parasitized"
#             confidence = (1 - prob) * 100
#         else:
#             pred_class = "Uninfected"
#             confidence = prob * 100

#         # Display results
#         st.success(f"🧪 Prediction: **{pred_class}**")
#         st.info(f"🔍 Confidence: **{confidence:.2f}%**")
#         st.markdown("<small>Note: This tool uses a deep learning model trained on malaria cell images.</small>", unsafe_allow_html=True)


# # Footer
# st.markdown("""
# ---
# Built by **OGUNSOLA, Oluwatosin**  
# Powered by **TensorFlow** & **Streamlit**  
# Helping improve malaria diagnostics through AI.
# """)



#import os
#os.environ["XDG_STATE_HOME"] = "/tmp"  # Set writable path for Streamlit
# Force Streamlit and other libraries to use your app folder
#os.environ["HOME"] = os.getcwd()
#os.environ["HF_HOME"] = os.getcwd()  # Also helps for Hugging Face caching
#os.makedirs(os.path.join(os.getcwd(), ".streamlit"), exist_ok=True)




# import streamlit as st
# import numpy as np
# import cv2 as cv
# from tensorflow.keras.models import load_model
# from PIL import Image

# # Load model
# model = load_model("malaria_model.tflite")  # Ensure this filename matches what you've uploaded

# # Class labels
# class_names = ['Parasitized', 'Uninfected']  # 0 = Parasitized, 1 = Uninfected

# # Streamlit page config
# st.set_page_config(page_title="Malaria Cell Classifier", layout="centered")
# st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧬 Malaria Cell Image Classifier</h1>", unsafe_allow_html=True)
# st.markdown("<p style='text-align: center;'>Upload a blood smear image to detect if the cell is infected with malaria or not.</p>", unsafe_allow_html=True)

# # Upload image
# uploaded_file = st.file_uploader("📤 Upload a cell image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

#     # Decode and preprocess image
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)

#     if image is None:
#         st.error("❌ Could not load the image.")
#     else:
#         img = cv.resize(image, (128, 128))  # Resize to model input size
#         img = img.astype("float32") / 255.0
#         img_array = np.expand_dims(img, axis=0)  # Shape (1, 128, 128, 3)

#         # Predict
#         prediction = model.predict(img_array)[0][0]

#         if prediction < 0.5:
#             pred_class = class_names[0]  # Parasitized
#             confidence = (1 - prediction) * 100
#         else:
#             pred_class = class_names[1]  # Uninfected
#             confidence = prediction * 100

#         # Display result
#         st.success(f"🧪 Prediction: **{pred_class}**")
#         st.info(f"🔍 Confidence: **{confidence:.2f}%**")
#         st.markdown("<small>Note: This tool uses a deep learning model trained on malaria cell images.</small>", unsafe_allow_html=True)

#         st.markdown("""
#         <small>
#          <b>Developer:</b> OGUNSOLA, Oluwatosin Adepeju  
#          <b>Role:</b> Data Scientist | AI Engineer Trainee @ Webfala Digital Skills for all Initiative 
#          <b>Note:</b> This project was developed during a Computer Vision workshop session organised by Women in AI Nigeria.
#         </small>
#         """, unsafe_allow_html=True)

import streamlit as st
import numpy as np
import cv2 as cv
import tensorflow as tf
from PIL import Image

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="malaria_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels
class_names = ['Parasitized', 'Uninfected']  # 0 = Parasitized, 1 = Uninfected

# Streamlit page config
st.set_page_config(page_title="Malaria Cell Classifier", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧬 Malaria Cell Image Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a blood smear image to detect if the cell is infected with malaria or not.</p>", unsafe_allow_html=True)

# Upload image
uploaded_file = st.file_uploader("📤 Upload a cell image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Decode and preprocess image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)

    if image is None:
        st.error("❌ Could not load the image.")
    else:
        img = cv.resize(image, (128, 128))  # Resize to match model input
        img = img.astype("float32") / 255.0
        img_array = np.expand_dims(img, axis=0)

        # TFLite requires float32 input
        img_array = img_array.astype(np.float32)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

        if prediction < 0.5:
            pred_class = class_names[0]  # Parasitized
            confidence = (1 - prediction) * 100
        else:
            pred_class = class_names[1]  # Uninfected
            confidence = prediction * 100

        # Display result
        st.success(f"🧪 Prediction: **{pred_class}**")
        st.info(f"🔍 Confidence: **{confidence:.2f}%**")
        st.markdown("<small>Note: This tool uses a lightweight TFLite model trained on malaria cell images.</small>", unsafe_allow_html=True)

        st.markdown("""
        <small>
         <b>Developer:</b> OGUNSOLA, Oluwatosin Adepeju  
         <b>Role:</b> Data Scientist | AI Engineer Trainee @ Webfala Digital Skills for All Initiative  
         <b>Note:</b> This project was developed during a Computer Vision workshop session organised by Women in AI Nigeria.
        </small>
        """, unsafe_allow_html=True)

