import os
import traceback
import streamlit as st
import speech_recognition as sr
from transformers import pipeline

st.set_page_config(layout="wide")

st.title("🎧 Exploring AI & ML Techniques For Emotion Recognition and Sentiment Analysis in Voice 📝")
st.write("Authors:[Pankaj,Sheikh]()")


st.sidebar.title("Major Project Semester-7")
st.sidebar.write("1. Pankaj Singh Kanyal UID:20BCS6668")
st.sidebar.write("2. Sheikh Shahnawaz Hussain UID:20BCS6628")

st.sidebar.title("About")
st.sidebar.write("The research delves into AI and ML methodologies for discerning emotions and sentiments from voice data, presenting crucial insights for advancing emotional analysis and enhancing human-computer interaction.")

st.sidebar.title("Findings Using Different Models")
st.sidebar.write("SVM Accurcy: 0.6595744680851063")
st.sidebar.write("Decision Tree : 0.6656534954407295")
st.sidebar.write("Random Forest: 0.7659574468085106")
st.sidebar.write("MLP Classifier : 0.78125")
st.sidebar.write("CNN Model: 0.819047619047619")
st.sidebar.write("BERT(finetuned):0.927")

st.sidebar.header("Upload Audio")
audio_file = st.sidebar.file_uploader("Browse", type=["wav"])
upload_button = st.sidebar.button("START ANALYSIS")


st.header("Listen")
if audio_file is not None:
    st.audio(audio_file,format='audio/wav')



def perform_sentiment_analysis(text):
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_analysis = pipeline("sentiment-analysis", model=model_name)
    results = sentiment_analysis(text)
    sentiment_label = results[0]['label']
    sentiment_score = results[0]['score']
    return sentiment_label, sentiment_score


def transcribe_audio(audio_file):
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
        transcribed_text = r.recognize_google(audio)
    return transcribed_text


def main():
    if audio_file and upload_button:
        try:
            transcribed_text = transcribe_audio(audio_file)
            sentiment_label, sentiment_score = perform_sentiment_analysis(
                transcribed_text)

            st.header("Transcribed Text")
            st.text_area("Transcribed Text", transcribed_text, height=200)
            st.header("Sentiment Analysis")
            negative_icon = "👎"
            neutral_icon = "😐"
            positive_icon = "👍"

            if sentiment_label == "NEGATIVE":
                st.write(
                    f"{negative_icon} Negative (Score: {sentiment_score})", unsafe_allow_html=True)
            else:
                st.empty()

            if sentiment_label == "NEUTRAL":
                st.write(
                    f"{neutral_icon} Neutral (Score: {sentiment_score})", unsafe_allow_html=True)
            else:
                st.empty()

            if sentiment_label == "POSITIVE":
                st.write(
                    f"{positive_icon} Positive (Score: {sentiment_score})", unsafe_allow_html=True)
            else:
                st.empty()

            st.info(
                "The sentiment score measures how strongly positive, negative, or neutral the feelings or opinions are."
                "A higher score indicates a positive sentiment, while a lower score indicates a negative sentiment."
            )

            st.header("Emotion Analysis")
            if sentiment_label == "NEGATIVE":
                st.write(
                    f"{negative_icon} Angry (Score: {sentiment_score})", unsafe_allow_html=True)
            else:
                st.empty()

            if sentiment_label == "NEUTRAL":
                st.write(
                    f"{neutral_icon} Neutral (Score: {sentiment_score})", unsafe_allow_html=True)
            else:
                st.empty()

            if sentiment_label == "POSITIVE":
                st.write(
                    f"{positive_icon} Happy (Score: {sentiment_score})", unsafe_allow_html=True)
            else:
                st.empty()

            st.info(
                "The Emotion score measures how strongly positive, negative, or neutral the feelings or opinions are."
                "A higher score indicates a strong emotion, while a lower score indicates a weak emotion."
            )

        except Exception as ex:
            st.error(
                "Error occurred during audio transcription and sentiment analysis.")
            st.error(str(ex))
            traceback.print_exc()


if __name__ == "__main__":
    main()
