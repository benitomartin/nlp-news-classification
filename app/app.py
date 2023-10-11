import pandas as pd
import streamlit as st
import tensorflow as tf
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# Text Preprocessing
def preprocess_text(text_series):
    # Initialize an empty list to store the preprocessed texts
    preprocessed_texts = []

    for text in text_series:
        # Remove punctuation
        for punctuation in string.punctuation:
            text = text.replace(punctuation, ' ')

        # Lowercase the text
        text = text.lower()

        # Tokenize the text
        tokenized = word_tokenize(text)

        # Remove non-alphabetic words
        words_only = [word for word in tokenized if word.isalpha()]

        # Remove stop words
        stop_words = set(stopwords.words('english'))
        without_stopwords = [word for word in words_only if word not in stop_words]

        # Lemmatize the words
        lemma = WordNetLemmatizer()
        lemmatized = [lemma.lemmatize(word) for word in without_stopwords]

        # Join the cleaned words back into a string
        cleaned_text = ' '.join(lemmatized)

        # Append the preprocessed text to the list
        preprocessed_texts.append(cleaned_text)

    return preprocessed_texts

# Prediction
def predict_fake_news(text, model):
    # Preprocess the text
    cleaned_text = preprocess_text(text)

    # Make predictions
    output = model.signatures['serving_default'](input_5=tf.constant([cleaned_text]))


    # Extract the output (you may need to adapt this based on your model's output)
    # For example, if your model outputs a single probability, you can do:
    fake_news_probability = output['dense_4'].numpy()[0]  # Replace 'dense_4' with the correct output name

    return fake_news_probability

def main():
    """ NLP Based App with Streamlit """

    # Set page title and configure layout
    st.set_page_config(page_title="Fake News Detection App", page_icon="📰", layout="wide")

    # Add a title and description
    st.title("Fake News Detection App")
    st.markdown("""
        #### Description
        The news detection uses the best trained model, a TensorFlow Bidirectional LSTM Model, which achieved an accuracy above 99.84 %. You can find the notebook with the different trained models in the following [repository](https://github.com/benitomartin/nlp-news-classification)
        """)

    # Add a heading for user input
    st.header("Input News Text")

    # Create a text area for user input
    message = st.text_area("🗞️🗞️🗞️ Enter the news text 🗞️🗞️🗞️", placeholder="Paste the news text here...")

    text_series = pd.Series([message])

    
    # Load your TensorFlow SavedModel
    model = tf.saved_model.load('model')  # Replace 'your_model_directory' with the actual directory containing your SavedModel


    if st.button("Predict"):
        fake_news_probability = predict_fake_news(text_series, model)

        # Determine whether it's True or Fake based on the threshold (0.5)
        result = "True" if fake_news_probability > 0.5 else "Fake"

        # Display the result in a styled div element
        if result == "True":
            st.success("This news is labeled as:")
        else:
            st.error("This news is labeled as:")

        # Display the result with a color-coded label
        st.write(f"<div style='font-size: 24px; text-align: center; color: {'green' if result == 'True' else 'red'};'>{result}</div>", unsafe_allow_html=True)

    st.sidebar.subheader("About the App")
    st.sidebar.info("This App is based on the following [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/data) dataset")
    st.sidebar.markdown("![Image](https://cdn.britannica.com/25/93825-050-D1300547/collection-newspapers.jpg)")


if __name__ == '__main__':
    main()
