# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 23:37:02 2023

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
import tweepy
import pickle

# Twitter API credentials
consumer_key = "YOUR_CONSUMER_KEY"
consumer_secret = "YOUR_CONSUMER_SECRET"
access_token = "YOUR_ACCESS_TOKEN"
access_token_secret = "YOUR_ACCESS_TOKEN_SECRET"

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Load the saved model and tokenizer
model = load_model("bot_detection_model.h5")
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

    # Create an input field for the Twitter username
    username = st.text_input('Enter Twitter username (e.g., @example)')

    # Check if the user has entered a username
    if username:
        try:
            # Fetch the user's recent tweets
            tweets = api.user_timeline(screen_name=username, count=10)

            # Extract the tweet texts
            tweet_texts = [tweet.text for tweet in tweets]

            # Preprocess the tweets
            processed_tweets = [preprocess_tweet(tweet) for tweet in tweet_texts]

            # Convert the processed tweets to sequences
            tweet_sequences = tokenizer.texts_to_sequences(processed_tweets)

            # Pad the sequences
            tweet_padded = pad_sequences(tweet_sequences, maxlen=max_sequence_length)

            # Make predictions using the loaded model
            predictions = model.predict(tweet_padded)
            predictions = np.round(predictions).flatten()

            st.subheader("Prediction results:")

            for tweet, prediction in zip(tweet_texts, predictions):
                st.write(f"Tweet: {tweet}")
                if prediction == 0:
                    st.write('Bot: No')
                else:
                    st.write('Bot: Yes')
                st.write("---")

        except tweepy.TweepError as e:
            st.error(f"Error: {e}")
    
# Run the app
if __name__ == '__main__':
    main()
