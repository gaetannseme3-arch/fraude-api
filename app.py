from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model_fraude.pkl")

@app.route("/")
def home():
    return "API de détection de fraude"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data["features"]
        input_data = np.array(features).reshape(1, -1)
        prediction = model.predict(input_data)
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)