from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load Model and Scaler
model = joblib.load("model/heart_model.pkl")
scaler = joblib.load("model/scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get user input
    data = [
        float(request.form["age"]),
        float(request.form["sex"]),
        float(request.form["cp"]),
        float(request.form["trestbps"]),
        float(request.form["chol"]),
        float(request.form["fbs"]),
        float(request.form["restecg"]),
        float(request.form["thalach"]),
        float(request.form["exang"]),
        float(request.form["oldpeak"]),
        float(request.form["slope"]),
        float(request.form["ca"]),
        float(request.form["thal"])
    ]

    # Scale the input
    data = scaler.transform([data])

    # Prediction
    prediction = model.predict(data)

    # Result
    if prediction[0] == 1:

        result = "❤️ Heart Disease Detected"
        risk = "🔴 HIGH RISK"

        recommendations = [
            "Consult a cardiologist immediately.",
            "Monitor your blood pressure regularly.",
            "Follow a healthy low-fat and low-salt diet.",
            "Exercise only according to your doctor's advice.",
            "Avoid smoking and alcohol.",
            "Schedule regular heart check-ups."
        ]

    else:

        result = "💚 No Heart Disease Detected"
        risk = "🟢 LOW RISK"

        recommendations = [
            "Maintain a healthy lifestyle.",
            "Exercise regularly.",
            "Eat a balanced diet.",
            "Maintain a healthy weight.",
            "Monitor blood pressure periodically.",
            "Continue routine medical check-ups."
        ]

    return render_template(
        "result.html",
        prediction=result,
        risk=risk,
        recommendations=recommendations,
        accuracy="77.0"
    )


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)