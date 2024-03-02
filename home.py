import streamlit as st
from model1 import predictor

st.title(" Accident Detection using machine learning")

uploaded_file = st.file_uploader("Upload your file here...", type=['png', 'jpeg', 'jpg'])

if uploaded_file is not None:
    st.image(uploaded_file)

try:
    st.write(predictor(uploaded_file))
except:
    st.warning("Please upload a file")
