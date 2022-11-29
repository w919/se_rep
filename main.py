from transformers import pipeline, set_seed
import streamlit as st
import numpy as np


txt = st.text_input('Text sample')

generator = pipeline('text-generation', model='gpt2')


st.title('Text generator gpt2')
length = st.slider('Max_length', 30, 120)
num_seq = st.slider('num_seq', 1, 3)
result = st.button('Get text')
output = generator(txt, max_length=length, num_return_sequences=num_seq)

if result:
    text = ''
    for i in range(len(output)):
       text += (output[i].get('generated_text') + '\n')
    st.text_area('Generated text', text)
