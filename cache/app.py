import streamlit as st
import nltk
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import pandas as pd

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('vader_lexicon')

# Function to perform sentiment analysis using TextBlob for each sentence
def analyze_sentiment(text):
    sentences = nltk.sent_tokenize(text)
    polarity_scores = []
    subjectivity_scores = []
    for sentence in sentences:
        blob = TextBlob(sentence)
        polarity_scores.append(blob.sentiment.polarity)
        subjectivity_scores.append(blob.sentiment.subjectivity)
    return polarity_scores, subjectivity_scores

# Function to perform sentiment analysis using NLTK's VaderSentimentAnalyzer for each sentence
def analyze_sentiment_nltk(text):
    sid = SentimentIntensityAnalyzer()
    sentences = nltk.sent_tokenize(text)
    scores = []
    for sentence in sentences:
        scores.append(sid.polarity_scores(sentence)['compound'])
    return scores

# Function to perform temporal analysis using NLTK
def temporal_analysis(text):
    sentences = sent_tokenize(text)
    words = [word.lower() for sentence in sentences for word in nltk.word_tokenize(sentence)]
    fdist = FreqDist(words)
    return fdist

# Sample paragraph
avm = "People may choose to go to other suitable private or trust hospitals if the CHC is far away. This is because not everyone can afford the time, energy, and expenses to travel to a distant CHC. In such cases, it may be more practical for people to go to other local hospitals that are closer to them. These could be private hospitals, trust hospitals, or community hospitals. It is important to know where these hospitals are located so that people can make informed decisions about where to seek medical help. Some options for locating nearby hospitals could be: referring the patient to the nearest PHC/CHC where facilities for admission are available, escorting the patient to the nearest CHC/PHC where facilities are available, or arranging for transport"

# Sentiment Analysis
polarity_scores, subjectivity_scores = analyze_sentiment(avm)
sentiment_scores_nltk = analyze_sentiment_nltk(avm)

# Subjectivity Analysis
subjectivity_scores = analyze_sentiment(avm)[1]

# Temporal Analysis
fdist = temporal_analysis(avm)
df = pd.DataFrame(list(fdist.items()), columns=['Word', 'Frequency']).sort_values(by='Frequency', ascending=False)

# Streamlit App
st.title('Sentiment and Temporal Analysis for Each Sentence')

# Arrange the graphs as 2 on top and 2 on bottom
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

# Sentiment Polarity for each sentence
axes[0, 0].bar(range(1, len(polarity_scores) + 1), polarity_scores)
axes[0, 0].set_title('Sentiment Polarity for Each Sentence')
axes[0, 0].set_xlabel('Sentence Number')
axes[0, 0].set_ylabel('Sentiment Polarity')

# Sentiment Strength for each sentence
axes[0, 1].bar(range(1, len(sentiment_scores_nltk) + 1), sentiment_scores_nltk)
axes[0, 1].set_title('Sentiment Strength for Each Sentence')
axes[0, 1].set_xlabel('Sentence Number')
axes[0, 1].set_ylabel('Sentiment Strength')

# Subjectivity Analysis for each sentence
axes[1, 0].bar(range(1, len(subjectivity_scores) + 1), subjectivity_scores)
axes[1, 0].set_title('Subjectivity Analysis for Each Sentence')
axes[1, 0].set_xlabel('Sentence Number')
axes[1, 0].set_ylabel('Subjectivity')

# Temporal Analysis for each sentence
axes[1, 1].bar(df['Word'][:10], df['Frequency'][:10])
axes[1, 1].set_title('Temporal Analysis (Top 10 Words)')

# Display the plots
st.pyplot(fig)
