import joblib
from src.utils import label_to_sentiment

def load_model():
    model = joblib.load("models/best_classifier.joblib")
    vectorizer = joblib.load("models/vectorizer.joblib")
    le = joblib.load("models/label_encoder.joblib")
    return model, vectorizer, le

def predict(text):
    model, vectorizer, le = load_model()
    
    # Vectorize the input text
    text_vec = vectorizer.transform([text])
    
    # Predict
    prediction = model.predict(text_vec)
    
    # Convert prediction to sentiment label
    sentiment = label_to_sentiment(prediction[0], le)
    
    return sentiment

if __name__ == "__main__":
    text = "Це був чудовий день, ми чудово провели час."
    print(f"Sentiment: {predict(text)}")