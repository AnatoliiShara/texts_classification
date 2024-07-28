from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.utils import load_data, preprocess_data, get_sentiment_distribution
import joblib
import numpy as np

def train():
    # Load and preprocess data
    data = load_data("data/sentiment_data.csv")
    X_train, X_test, y_train, y_test, vectorizer, le = preprocess_data(data)

    # Print class distribution
    print("Training set class distribution:")
    print(get_sentiment_distribution(y_train))
    print("Test set class distribution:")
    print(get_sentiment_distribution(y_test))

    # Train Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    nb_accuracy = accuracy_score(y_test, nb_pred)
    print("Naive Bayes Accuracy:", nb_accuracy)
    print("Naive Bayes Classification Report:\n", classification_report(y_test, nb_pred, target_names=le.classes_))

    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    print("Random Forest Accuracy:", rf_accuracy)
    print("Random Forest Classification Report:\n", classification_report(y_test, rf_pred, target_names=le.classes_))

    # Train LightGBM
    lgb_model = LGBMClassifier(random_state=42)
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)
    lgb_accuracy = accuracy_score(y_test, lgb_pred)
    print("LightGBM Accuracy:", lgb_accuracy)
    print("LightGBM Classification Report:\n", classification_report(y_test, lgb_pred, target_names=le.classes_))

    # Save the best model based on accuracy
    best_model = max([
        (nb_model, nb_accuracy),
        (rf_model, rf_accuracy),
        (lgb_model, lgb_accuracy)
    ], key=lambda x: x[1])[0]

    joblib.dump(best_model, "models/best_classifier.joblib")
    joblib.dump(vectorizer, "models/vectorizer.joblib")
    joblib.dump(le, "models/label_encoder.joblib")

if __name__ == "__main__":
    train()