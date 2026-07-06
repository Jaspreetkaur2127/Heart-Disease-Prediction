from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Model aur Scaler Load
model = joblib.load("model/heart_model.pkl")
scaler = joblib.load("model/scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

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

    data = scaler.transform([data])

    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "Heart Disease Detected"
    else:
        result = "No Heart Disease"

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)