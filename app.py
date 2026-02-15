import streamlit as st
from PIL import Image
from utils import predict_dog_breed

st.set_page_config(page_title="🐶 Dog Breed Detector")
st.title("🐕 Dog Breed Detector with PyTorch")

uploaded_file = st.file_uploader("Upload a dog image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect Breed"):
        with st.spinner("Predicting..."):
            breed, confidence = predict_dog_breed(image)
            st.success(f"Breed: **{breed}**\n\nConfidence: **{confidence}%**")
