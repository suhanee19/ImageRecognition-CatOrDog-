import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image

model = load_model("cat_dog_model.h5")

st.title("🐱🐶 Cat vs Dog Classifier")
st.write("Upload an image, and I'll tell you if it's a cat or a dog.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    label = "Dog 🐶" if prediction[0][0] > 0.5 else "Cat 🐱"

    st.markdown(f"### Prediction: **{label}**")
