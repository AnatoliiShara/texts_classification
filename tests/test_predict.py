import unittest
from src.predict import predict

class TestPredict(unittest.TestCase):
    def test_predict_positive(self):
        text = "Це був чудовий день, ми чудово провели час."
        sentiment = predict(text)
        self.assertEqual(sentiment, 'positive')

    def test_predict_negative(self):
        text = "Це був жахливий день, все пішло не так."
        sentiment = predict(text)
        self.assertEqual(sentiment, 'negative')

    def test_predict_neutral(self):
        text = "Сьогодні звичайний день, нічого особливого."
        sentiment = predict(text)
        self.assertEqual(sentiment, 'neutral')

    def test_predict_neutral_positive(self):
        text = "День був в цілому непоганий, але без чогось особливого."
        sentiment = predict(text)
        self.assertEqual(sentiment, 'neutral_positive')

    def test_predict_negative_neutral(self):
        text = "День був поганим, але деякі моменти були приємними."
        sentiment = predict(text)
        self.assertEqual(sentiment, 'negative_neutral')

if __name__ == '__main__':
    unittest.main()