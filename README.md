# **🦴 Disease Prediction for Osteoporosis**

This project is a machine learning–based web application that predicts Osteoporosis conditions from bone X-ray images. 
The application is built using Streamlit and a trained Logistic Regression model, allowing users to upload X-ray images and receive instant predictions.


**Project Overview**

Osteoporosis is a bone disease that leads to decreased bone density and increased fracture risk. Early detection plays a crucial role in preventing severe complications.


**This application:**

- Accepts bone X-ray images

- Validates whether the uploaded image is a valid X-ray

- Predicts the disease condition using a trained ML model



  The model predicts one of the following categories:

- Normal

- Osteopenia

- Osteoporosis



  **Features :**

📤 Upload bone X-ray images (JPG, PNG, JPEG)

🧪 Image preprocessing and validation

❌ Error handling for non-X-ray images

🎯 Accurate prediction using ML model

🎨 Interactive UI with custom background and styling


**Technologies Used :**

- Python

- Streamlit

- NumPy

- Pillow (PIL)

- Scikit-learn

- Logistic Regression

- Pickle


**Project Structure:**

  Disease-Prediction-for-Osteoporosis/
  
│
├── app.py                 # Streamlit application

├── logi_reg.pkl           # Trained ML model

├── requirements.txt       # Project dependencies

├── background.png         # UI background image

└── README.md              # Project documentation



**Model Details**

Algorithm: Logistic Regression

Input: Flattened image pixel values

Output: Disease class label

Model Storage: logi_reg.pkl
