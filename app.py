import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="banana_leaf_multiclass.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels (MUST match class_indices)
labels = [
    "Cordana",
    "Healthy",
    "Panama Disease",
    "Sigatoka"
]

st.set_page_config(page_title="Banana Leaf Disease Detector", layout="centered")

st.title("🍌 Banana Leaf Disease Detection")
st.write("Upload a banana leaf image to detect the disease.")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    idx = np.argmax(preds)
    disease = labels[idx]
    confidence = preds[idx] * 100

    st.subheader("🧪 Diagnosis")
    st.success(f"**{disease}**")
    st.write(f"Confidence: **{confidence:.2f}%**")
