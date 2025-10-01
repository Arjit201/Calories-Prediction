from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS   # add this

app = Flask(__name__)
CORS(app)   # enable CORS

model = joblib.load("calorie_predictor.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array([
        data["gender"],
        data["age"],
        data["height"],
        data["weight"],
        data["duration"],
        data["heart_rate"],
        data["body_temp"]
    ]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return jsonify({"calories_burnt": float(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
