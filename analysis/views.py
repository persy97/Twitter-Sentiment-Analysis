from django.shortcuts import render, HttpResponse
import tweepy
from tweepy import OAuthHandler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Create your views here.
consumer_key = 	"#Enter your Consumer Key"
consumer_secret = "#Enter your Consumer Secret"
access_token = "#Enter Access Key"
access_token_secret = "#Enter Access Secret"


datasets = pd.read_csv('C:\\Users\\HP\\Desktop\\Edited.csv', encoding='latin-1', header=None)

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
mystopwords = set(stopwords.words('English')) - set('not')
for i in datasets[1]:
        result = re.sub(r"http\S+", " ", i)
        result = re.sub(r"(@[A-Za-z0-9]+)", " ", result)
        result = re.sub(r"[^a-zA-z]", " ", result)
        result = result.lower().split()
        result = [ps.stem(word) for word in result if not word in mystopwords]
        result = ' '.join(result)
        corpus.append(result)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1200)
X = cv.fit_transform(corpus).toarray()
y = datasets.iloc[:, 0].values



from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X, y)


def home(request):
    return render(request, 'homepage/homepage.html')


def analysis(request):
    tweet_query = request.GET.get('topic')
    tweet_count = request.GET.get('number')
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    tweets = []
    fetched_tweets = api.search(q=tweet_query, count=tweet_count)
    for tweet in fetched_tweets:
        tweets.append(tweet.text)

    cleaned_tweets = []
    for i in tweets:
        result = re.sub(r"http\S+", " ", i)
        result = re.sub(r"(@[A-Za-z0-9]+)", " ", result)
        result = re.sub(r"[^a-zA-z]", " ", result)
        result = result.lower().split()
        result = [ps.stem(word) for word in result if not word in mystopwords]
        result = ' '.join(result)
        cleaned_tweets.append(result)

    a = cv.transform(cleaned_tweets).toarray()
    a = classifier.predict(a)
    positive = 0
    positive_tweets = []
    negative_tweets = []
    c = 0
    for k in a:
        c = c + 1

        if k == 4:
            positive_tweets.append(tweets[c-1])
            positive = positive + 1
        else:
            negative_tweets.append(tweets[c-1])

    positive = (positive / len(a)) * 100
    positive = int(positive)
    negative = 100 - float(positive)
    print(positive)
    print(negative)

    context = {"topic": tweet_query, "positive_tweets": positive_tweets,
               "negative_tweets": negative_tweets,
               "positive_percentage": str(positive),
               "negative_percentage": str(negative),
    }
    return render(request, 'result/result.html', context)
