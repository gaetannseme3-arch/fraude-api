from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = None

def load_model():
    global model
    if model is None:
        model = joblib.load("model_fraude.pkl")
    return model


@app.route("/")
def home():
    return "API fraude active"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({"error": "Aucune donnée JSON reçue"}), 400

        expected_columns = [
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest"
        ]

        for col in expected_columns:
            if col not in data:
                data[col] = 0

        df = pd.DataFrame([data])
        modele = load_model()
        prediction = modele.predict(df)

        return jsonify({
            "prediction": int(prediction[0])
        })

    except Exception as e:
        import traceback
        return jsonify({
            "error": repr(e),
            "trace": traceback,format_exc()
            }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)