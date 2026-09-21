import re

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st


# Must match the values used when training the model
MAX_FEATURES = 10000   # vocabulary size used in training
MAX_LENGTH = 500       # sequence length used in training
MODEL_PATH = "simple_rnn_imdb.h5"


# Load the IMDB word index and the trained model once (cached between reruns)
@st.cache_resource
def load_resources():
    word_index = imdb.get_word_index()
    model = load_model(MODEL_PATH)
    return word_index, model


word_index, model = load_resources()


# Convert the review into numbers and pad it
def preprocess_text(text):
    # Keep only letters and apostrophes so "great!" becomes "great"
    words = re.findall(r"[a-z']+", text.lower())

    encoded_review = []
    for word in words:
        idx = word_index.get(word)

        # Unknown or out-of-vocabulary words map to the "unknown" token (2)
        if idx is None or idx + 3 >= MAX_FEATURES:
            encoded_review.append(2)
        else:
            # Keras IMDB reserves indices 0-3, so real words are shifted by 3
            encoded_review.append(idx + 3)

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=MAX_LENGTH
    )

    return padded_review


# Get the sentiment from the model
def predict_sentiment(review):
    preprocessed_review = preprocess_text(review)

    prediction = model.predict(
        preprocessed_review,
        verbose=0
    )

    score = float(prediction[0][0])
    sentiment = "Positive" if score > 0.35 else "Negative"

    return sentiment, score


# Show the result card.
# The HTML has NO blank lines and NO indentation on purpose:
# Markdown would otherwise turn indented lines into a code block.
def show_result(sentiment, score):
    emoji, label = (
        ("😊", "POSITIVE") if sentiment == "Positive" else ("😞", "NEGATIVE")
    )

    html = (
        '<div class="result-card">'
        '<div class="result-title">🎯 Predicted Sentiment</div>'
        f'<div class="sentiment">{emoji} {label}</div>'
        f'<div class="score">Model Score: <b>{score:.4f}</b></div>'
        '</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


# Page settings
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)


# Some simple styling
st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #0f172a,
            #1e293b,
            #312e81
        );
        color: white;
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        margin-top: 20px;
        margin-bottom: 5px;

        background: linear-gradient(
            90deg,
            #38bdf8,
            #818cf8,
            #c084fc
        );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #cbd5e1;
        margin-bottom: 30px;
    }

    textarea {
        background-color: #1e293b !important;
        color: white !important;
        border: 2px solid #6366f1 !important;
        border-radius: 12px !important;
        font-size: 16px !important;
    }

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: none;
        padding: 12px;
        font-size: 18px;
        font-weight: 700;

        background: linear-gradient(
            90deg,
            #6366f1,
            #8b5cf6
        );

        color: white;
    }

    .result-card {
        margin-top: 25px;
        padding: 25px;
        border-radius: 18px;
        text-align: center;

        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);

        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
    }

    .result-title {
        font-size: 20px;
        color: #cbd5e1;
    }

    .sentiment {
        font-size: 36px;
        font-weight: 800;
        margin: 10px;
    }

    .score {
        font-size: 18px;
        color: #cbd5e1;
    }

    .footer {
        text-align: center;
        margin-top: 40px;
        color: #94a3b8;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# App heading
st.markdown(
    '<div class="main-title">🎬 IMDB Movie Sentiment Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    '🤖 Powered by a Simple RNN Deep Learning Model'
    '</div>',
    unsafe_allow_html=True
)


# Get the review from the user
st.markdown("### ✍️ Enter your movie review")

user_input = st.text_area(
    "Movie Review",
    placeholder="Example: This movie was absolutely amazing!",
    height=180,
    label_visibility="collapsed"
)


# Run the model when the button is clicked
if st.button("🔍 Analyze Sentiment"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter a movie review first.")

    else:
        sentiment, score = predict_sentiment(user_input)
        show_result(sentiment, score)


# Small footer
st.markdown(
    '<div class="footer">'
    '🧠 Simple RNN • IMDB Dataset • Sentiment Classification'
    '</div>',
    unsafe_allow_html=True
)