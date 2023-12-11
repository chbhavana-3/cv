#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
import pandas as pd
from transformers import pipeline

# Load the sentiment analysis model
emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')



# Function to get emotion label
def get_emotion_label(text):
    return emotion(text)[0]['label']

# Apply the function to the dataset
#sin['emotion'] = sin['text'].apply(get_emotion_label)

# Streamlit UI
st.title("Text Emotion Analyzer")

# Display the dataset
#st.dataframe(sin)

# Display emotion distribution
st.title("Emotion Distribution")
#st.bar_chart(sin['emotion'].value_counts())

# Allow users to input text and get the predicted emotion
user_input = st.text_area("Enter your text:")
if user_input:
    user_emotion = get_emotion_label(user_input)
    st.write(f"Predicted Emotion: {user_emotion}")


# In[ ]:




