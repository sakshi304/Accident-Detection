import streamlit as st
from predictor import predict

st.title(" Accident Detection using machine learning")

uploaded_file = st.file_uploader("Upload your file here...", type=['png', 'jpeg', 'jpg'])

try:
    st.write(predict(uploaded_file))
except:
    st.warning("Please upload a file")

if uploaded_file is not None:
    st.image(uploaded_file)
