import os
import numpy as np
import tensorflow as tf
from src.pipelines.prediction_pipeline import predict_next_word
from tensorflow.keras.models import load_model
from src.utils import load_object
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

file_path = os.path.join(os.getcwd(),'artifacts','model_path.txt')

with open(file_path,'r') as file:
    lines = file.readlines()

tokenizer_path =  lines[0].split("==")[0]
model_path =  lines[0].split("==")[1]

tokenizer_path = os.path.join(os.getcwd(),'models',tokenizer_path)
model_path  = os.path.join(os.getcwd(),'models',model_path)

tokenizer = load_object(tokenizer_path)
model = load_model(model_path)
max_sequence_len = model.input_shape[1]+1

st.markdown("<h1 style='text-align: center'>📚 ProsePredict: Advanced LSTM & GRU Text Forecasting</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("📊 Model Training Data:")
    st.write("The current model is trained on the following data:")
    
    data_folder_path = os.path.join(os.getcwd(),'data')
    all_files = os.listdir(data_folder_path)
    text_file = [file for file in all_files if file.endswith('.txt')]

    text_data_file_path = os.path.join(data_folder_path,text_file[0])

    with open(text_data_file_path,'r',encoding='utf-8') as file:
        text = file.read()
    st.text_area("Model Training Data:", value= text, height=500)

st.markdown("<h3 style='text-align: left'>📝 Type Your Text:</h3>", unsafe_allow_html=True)
input_text = st.text_area("", height=200)

predict_button = st.button("Predict Next Word")
predicted_word = ""

if predict_button:
    if len(input_text.split()) < 5:
        st.error("Please type at least 5 words to begin prediction.")
    else:
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        predicted_word = next_word
        st.text_input("Predicted Next Word:", value=predicted_word, disabled=True)

st.markdown("<h4 style='text-align: center'> 🚀 Customize Your Own ProsePredict Model!</h4>",unsafe_allow_html=True)
st.markdown("<p style='text-align: center'>This project is fully customizable. Follow a few simple steps to train your own model on your data for next word prediction. Want to know how? Check out the instructions on <a href='https://github.com/Aman-Vishwakarma1729/PDFy-chat-with-pdf' target='_blank'><b>GitHub</b></a>.</p>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center'>© 2024 Aman Vishwakarma 👨</h3>",unsafe_allow_html=True)



