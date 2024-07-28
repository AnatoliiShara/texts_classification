import unittest
import os
from src.train import train
from src.utils import load_data, preprocess_data

class TestTrain(unittest.TestCase):
    def test_train_creates_model_files(self):
        train()
        self.assertTrue(os.path.exists("models/best_classifier.joblib"))
        self.assertTrue(os.path.exists("models/vectorizer.joblib"))
        self.assertTrue(os.path.exists("models/label_encoder.joblib"))

    def test_data_loading_and_preprocessing(self):
        data = load_data("data/sentiment_data.csv")
        X_train, X_test, y_train, y_test, vectorizer, le = preprocess_data(data)
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(y_test)
        self.assertIsNotNone(vectorizer)
        self.assertIsNotNone(le)

if __name__ == '__main__':
    unittest.main()