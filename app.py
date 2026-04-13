from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Le modèle ne sera chargé qu'au moment de la prédiction
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

        # Si l'utilisateur envoie un seul objet JSON
        df = pd.DataFrame([data])

        # Charger le modèle seulement ici
        modele = load_model()

        prediction = modele.predict(df)

        return jsonify({
            "prediction": prediction[0].item() if hasattr(prediction[0], "item") else prediction[0]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)