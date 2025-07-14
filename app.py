from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Cargar modelo, escalador y PCA
model = joblib.load("modelo_completo_sin_rfe.pkl")
scaler = joblib.load("scaler_sin_rfe.pkl")
pca = joblib.load("pca_sin_rfe.pkl")

# Lista de nombres de features en el orden esperado
feature_order = [
    "Pclass", "Age", "SibSp", "Parch", "Fare",
    "Sex_male", "Embarked_Q", "Embarked_S", "HasCabin"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Extraer características en orden esperado
        feats = np.array([[float(data[feature]) for feature in feature_order]])

        # Transformar los datos
        feats_scaled = scaler.transform(feats)
        feats_pca = pca.transform(feats_scaled)

        # Predecir
        pred = model.predict(feats_pca)[0]
        return jsonify({"prediction": int(pred)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
