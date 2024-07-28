import flask
from flask import Flask, request, jsonify
from src.predict import predict

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.json
    text = data['text']
    sentiment = predict(text)
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
