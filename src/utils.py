import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

def load_data(filename):
    data = pd.read_csv(filename, encoding='utf-8')
    return data

def preprocess_data(data):
    X = data['text']
    y = data['sentiment']

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Vectorize the text
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec, y_train, y_test, vectorizer, le

def sentiment_to_label(sentiment, le):
    return le.transform([sentiment])[0]

def label_to_sentiment(label, le):
    return le.inverse_transform([label])[0]

def get_sentiment_distribution(y):
    unique, counts = np.unique(y, return_counts=True)
    return dict(zip(unique, counts))