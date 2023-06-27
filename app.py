# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 23:21:03 2023

@author: debna
"""

import streamlit as st
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import emoji  # Import the emoji library
nltk.download('stopwords')
nltk.download('wordnet')

# Load the saved model and tokenizer
model = load_model("T.h1")
tokenizer_path = "tokenizer.pickle"

with open(tokenizer_path, "rb") as file:
    tokenizer = pickle.load(file)

max_sequence_length = model.input_shape[1]

# Preprocess function
def preprocess_tweet(tweet):
    # Remove URLs
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)
    # Remove special characters and digits
    tweet = re.sub(r"[^\w\s]", "", tweet)
    tweet = re.sub(r"\d+", "", tweet)
    # Convert to lowercase
    tweet = tweet.lower()
    # Remove stop words and lemmatize words
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    tweet = " ".join([lemmatizer.lemmatize(word) for word in tweet.split() if word.lower() not in stop_words])
    return tweet

# Define the Streamlit app
def main():
    # Set the app title
    st.title('Twitter Bot Detection')

    # Create an input field for the user data
    user_data = st.text_input('Enter user data (e.g., profile information, tweets, etc.)')

    # Check if the user has entered any data
    if user_data:
        # Preprocess the user data
        processed_data = preprocess_tweet(user_data)

        # Convert the processed data to sequences
        user_sequence = tokenizer.texts_to_sequences([processed_data])

        # Pad the sequence
        user_padded = pad_sequences(user_sequence, maxlen=max_sequence_length)

        # Make predictions using the loaded model
        prediction = model.predict(user_padded)
        prediction = np.round(prediction).flatten()

        if prediction == 0:
            st.write('The user is Human. ' + emoji.emojize("\U0001F468", use_aliases=False))
        else:
            st.write('The user is a bot. ' + emoji.emojize(":robot_face:", use_aliases=True))

# Run the app
if __name__ == '__main__':
    main()
