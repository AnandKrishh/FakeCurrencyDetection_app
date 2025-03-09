import streamlit as st
import gdown
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image

# Set page config
st.set_page_config(page_title="Fake or Real Image Classifier", page_icon="\U0001F911", layout="wide", initial_sidebar_state="expanded")

# Set model input size
height = 300
width = 300

# Function to build the fine-tuned model
def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        x = Dense(fc, activation='relu')(x)
        x = Dropout(dropout)(x)
    
    predictions = Dense(num_classes, activation='softmax')(x)

    finetune_model = Model(inputs=base_model.input, outputs=predictions) 
    return finetune_model

# Use caching for loading the model to avoid repeated loading
@st.cache_resource
def load_model():
    # Load the pre-trained ResNet50 model (ensure consistent input size)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(height, width, 3))

    # Download the model weights from Google Drive (ensure download only once)
    gdown.download('https://drive.google.com/uc?id=1xeyNYt6KBw6yG4PvxAMqdHLe-cRYd656&export=download', 'Final_model.h5', quiet=False)

    # Rebuild the fine-tune model
    model = build_finetune_model(base_model, dropout=0.5, fc_layers=[1024, 1024], num_classes=2)

    # Try to load weights (use by_name=True if there's a mismatch in layers)
    try:
        model.load_weights("Final_model.h5", by_name=True)
        st.write("Model weights loaded successfully.")
    except Exception as e:
        st.write(f"Error loading model weights: {e}")
    return model

# Load the model once at the start
model = load_model()

# Class labels
class_list = ['Fake', 'Real']

# Prediction function
def predict_image(img, height, width):
    # Ensure the uploaded image is a PIL image
    if isinstance(img, Image.Image):
        img = img.resize((height, width))  # Resize to fit model input size
    else:
        raise ValueError("Input image is not a valid PIL image")

    # Convert the image to a numpy array and preprocess it
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Model prediction
    prediction = model.predict(img_array)
    predicted_class = class_list[np.argmax(prediction)]
    return predicted_class


# Streamlit app layout
st.markdown("""
    <style>
        .title {
            font-size: 60px;  
            color: #FF6347; 
            font-family: 'Albertus Extra Bold'; 
            font-weight: bold; 
            text-align: center; 
        }
    </style>
    <h1 class="title">Fake or Real Image Classifier</h1>
""", unsafe_allow_html=True)

st.write("Upload an image of either a real or a fake object, and the model will classify it.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Initialize variable to store the result
result = None

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Add a Submit button to trigger the analysis
    submit_button = st.button("Submit for Prediction")

    if submit_button:
        # Make prediction
        with st.spinner("Classifying..."):
            result = predict_image(img, height, width)  # Pass height and width as arguments
        
        # Display result
        st.write(f"Prediction: **{result}**")

        # Provide feedback or additional information
        if result == 'Real':
            st.success("The image is classified as Real!")
        else:
            st.error("The image is classified as Fake!")
