#  Malaria Cell Image Classifier

This is a Streamlit web app that classifies whether a given cell image is **Parasitized** (infected with malaria) or **Uninfected** using a Convolutional Neural Network (CNN) model trained on the NIH Malaria Dataset.

## How it Works
- Upload a blood smear image (JPG/PNG).
- The model processes the image and predicts if it's parasitized or uninfected.
- A **confidence score** is also displayed to show how certain the model is.

##  Developer Info
**Developed by OGUNSOLA, Oluwatosin Adepeju**, a Data Scientist and AI Engineer Trainee at Webfala Digital Skills for All Initiative.  
This project was built during a workshop session organized by **Women in AI Nigeria**, where she received hands-on training in **Computer Vision**.  
It showcases the application of AI in improving malaria diagnosis through automated blood smear classification.

##  Technologies Used
- Python 
- TensorFlow / Keras
- OpenCV
- Streamlit
- Hugging Face Spaces

##  Model Information
The model is a CNN trained on 27,000+ cell images from the **NIH Malaria Dataset**.  
It has been optimized for binary classification (Parasitized vs. Uninfected) and demonstrates strong performance.

##  Usage Instructions
1. Click on "Upload Image" to select a blood smear image.
2. Wait for the prediction to be displayed.
3. View the predicted class and confidence percentage.

##  Files Included
- `app.py` – The Streamlit app code
- `malaria_cnn_model.keras` – The trained CNN model file
- `requirements.txt` – Required dependencies
- `README.md` – This file with project info

##  Impact Statement
This model aids in **automated malaria detection**, potentially speeding up diagnosis and reducing human error in microscopy-based tests.  
It is part of a broader effort to integrate AI into global health solutions, especially across Africa.

##  Acknowledgement
Dataset: NIH Malaria Dataset  
Source: [https://lhncbc.nlm.nih.gov/publication/pub9932](https://lhncbc.nlm.nih.gov/publication/pub9932)
