import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
import google.generativeai as genai
import PIL.Image
import os


def load_xception_model(model_path):
    # Define the shape of the input images
    img_shape = (299,299, 3)

    # Load the Xception model pre-trained on ImageNet, excluding the top layer
    base_model = tf.keras.applications.Xception(include_top=False,
                                                weights="imagenet",
                                                input_shape=img_shape,
                                                pooling='max')

    # Create a Sequential model
    model = Sequential([
        base_model,  # Add the base model
        Flatten(),  # Flatten the output of the base model
        Dropout(rate=0.3),  # Add a dropout layer with a rate of 0.3
        Dense(128, activation='relu'),  # Add a dense layer with 128 units and ReLU activation
        Dropout(rate=0.25),  # Add another dropout layer with a rate of 0.25
        Dense(4, activation='softmax')  # Add a dense layer with 4 units and softmax activation for classification
    ])
    model.compile(Adamax(learning_rate=0.001), loss="categorical_crossentropy",
                  metrics=["accuracy", Precision(), Recall()])
    model.load_weights(model_path)
    
    return model

st.title("Brain Tumor Classification")
st.write("Upload an image of a brian MRI scan to classify")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    selected_model = st.radio(
        "Select Model",
        ("Transfer Learning - Xception", "Custom CNN")
    )
    
    if selected_model == "Transfer Learning - Xception":
        model = load_xception_model("../weights/xception_model.weights.h5")
        img_size = (299, 299)
    else:
        model = load_model("../weights/custom_model.weights.h5")
        img_size = (224, 224)
