import streamlit as st
from transformers import pipeline

# 1. Page Configuration
st.set_page_config(page_title="Sentiment Analyzer", page_icon="😊")

# 2. Cache the model to prevent reloading on every interaction
@st.cache_resource
def load_model():
    return pipeline('sentiment-analysis')

sentiment_pipeline = load_model()

# 3. User Interface
st.title("Simple Sentiment Analysis")
st.write("Enter text below to find out if the sentiment is Positive or Negative.")

user_input = st.text_area("Enter your text:", "I like this product")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # 4. Perform Analysis
        result = sentiment_pipeline(user_input)
        
        label = result[0]['label']
        score = result[0]['score']

        # 5. Display Results
        if label == "POSITIVE":
            st.success(f"Result: {label} (Confidence: {score:.2%})")
        else:
            st.error(f"Result: {label} (Confidence: {score:.2%})")
