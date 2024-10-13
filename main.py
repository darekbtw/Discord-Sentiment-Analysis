import requests
import json
import os
from dotenv import load_dotenv
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import streamlit as st

load_dotenv()
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def get_messages(channelId):
    headers = {
    'authorization': os.environ.get('authorization')
}

    r = requests.get(f'https://discord.com/api/v9/channels/{channelId}/messages', headers=headers)
    data = json.loads(r.text)
    res = []

    for i in data:
        res.append(i['content'])
    
    return res

def get_sentiment(compound):
    if compound >= 0.05:
        sentiment = 'Positive'
    elif compound <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    return sentiment

def analyze_messages(messages):
    total_score, total = 0, 0
    results = []

    for message in messages:
        sentiment = sia.polarity_scores(message)
        compound = sentiment['compound']
        curr_sentiment = get_sentiment(compound)
        total_score += compound
        total += 1

        result = f"\"{message}\" is {curr_sentiment}."
        results.append(result)
    
    avg_sentiment = float(total_score) / total
    overall_sentiment = get_sentiment(avg_sentiment)
    results.append(f"\nOverall Sentiment: {overall_sentiment}\n")
    return '\n'.join(results)



st.set_page_config(layout="wide", page_title="Discord Sentiment Analysis", page_icon="📰")
st.title('Discord Sentiment Analysis')
id_input = st.text_input("Enter a Discord Channel ID:", value="1073810997552349256")
submit_button = st.button("Submit")

if submit_button:
    try:
        messages = get_messages(id_input)
        res = analyze_messages(messages)

        st.code(res)
    except Exception as e:
        st.error(f"An Error Occurred: {e}")

