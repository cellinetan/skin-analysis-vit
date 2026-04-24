from flask import Flask, request, jsonify
import random

app = Flask(__name__)

@app.route('/')
def home():
    return "API is running"

@app.route('/predict', methods=['POST'])
def predict():
    result = {
        "forehead": random.choice(["Dry", "Normal"]),
        "nose": random.choice(["Oily", "Normal"]),
        "cheeks": random.choice(["Acne", "Normal"])
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)