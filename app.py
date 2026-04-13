@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({"error": "Aucune donnée JSON reçue"}), 400

        # ⚠️ colonnes attendues par le modèle
        expected_columns = [
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest"
        ]

        # compléter les colonnes manquantes avec 0
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
        return jsonify({"error": str(e)}), 500