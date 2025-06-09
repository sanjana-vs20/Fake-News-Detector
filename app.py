import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf # Import tensorflow for TFLite interpreter

# Load your Teachable Machine TFLite model and labels
@st.cache_resource # Cache the model to avoid reloading on every interaction
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    with open('labels.txt', 'r') as f:
        class_names = [name.strip() for name in f.readlines()] # Remove newline characters

    return interpreter, input_details, output_details, class_names

interpreter, input_details, output_details, class_names = load_tflite_model()

# Get expected input size (e.g., [1, 224, 224, 3])
input_shape = input_details[0]['shape']
IMG_HEIGHT = input_shape[1]
IMG_WIDTH = input_shape[2]

# Streamlit UI
st.title("Fake News Detector")
st.write("Upload an image of news text to detect if it's fake or real.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB") # Ensure image is in RGB
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Resize the image to the model's expected input size
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))

    # Convert image to numpy array and normalize
    # Teachable Machine's TFLite models typically expect float32 inputs normalized to -1 to 1
    # or uint8 inputs (0-255). Check your model's input_details[0]['dtype']
    input_data = np.asarray(image, dtype=np.float32) # Convert to float32
    input_data = (input_data / 127.5) - 1 # Normalize to -1 to 1 range

    # Add batch dimension
    input_data = np.expand_dims(input_data, axis=0) # Reshape to (1, IMG_HEIGHT, IMG_WIDTH, 3)

    # Set the tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the prediction results
    predictions = interpreter.get_tensor(output_details[0]['index'])

    # Process results
    index = np.argmax(predictions[0]) # Get the index of the highest probability
    class_name = class_names[index]
    confidence_score = predictions[0][index]

    st.write(f"Prediction: **{class_name}**")
    st.write(f"Confidence: **{confidence_score*100:.2f}%**")