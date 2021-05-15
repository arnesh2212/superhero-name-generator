import streamlit as st
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import io
import json

st.title("Neural Fictional Hero Name Generator By Arnesh")
st.write("Type the starting letter or letters and then press enter ( eg- a) :")
user_input = st.text_input("Enter Seed Text")


with open("superheroes.txt" , "r") as f:
    data = f.read()

with open('tokenizer.json') as f:
    data1 = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data1)
    



char_to_index = tokenizer.word_index
index_to_char = dict((v,k) for k,v in char_to_index.items())

names = data.splitlines()

def name_to_seq(name):
    return [tokenizer.texts_to_sequences(c)[0][0] for c in name]
def seq_to_name(seq):
    return [''.join([index_to_char[i] for i in seq if i !=0])]




model = keras.models.load_model("model.h5")
def generate_names(seed):
    for i in range(1,40):
        seq = name_to_seq(seed)
        padded = keras.preprocessing.sequence.pad_sequences([seq] , maxlen = 32 , padding = 'pre')
        pred =  model.predict(padded)[0]
        pred_char = index_to_char[np.argmax(pred)]
        seed += pred_char
        
        if pred_char == '\t':
            break
    return seed

name = generate_names(user_input)
st.write(name)