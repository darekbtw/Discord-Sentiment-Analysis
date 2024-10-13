import requests
import json
import os
from dotenv import load_dotenv
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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

    for message in messages:
        sentiment = sia.polarity_scores(message)
        compound = sentiment['compound']
        curr_sentiment = get_sentiment(compound)
        total_score+=compound
        total+=1

        print(f"\"{message}\" is {curr_sentiment}.")
    
    avg_sentiment = float(total_score) / total
    overall_sentiment = get_sentiment(avg_sentiment)
    print(f"\nSentiment: {overall_sentiment}\n")


messages = get_messages(1029181355017379900)
analyze_messages(messages)