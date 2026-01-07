from flask import Flask, render_template, request
import numpy as np
import joblib
from utils import extract_hrv_features

app = Flask(__name__)

svm = joblib.load("model/svm.pkl")
rf = joblib.load("model/rf.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("ecg_file")

    if not file:
        return render_template("index.html")

    ecg_signal = np.loadtxt(file)

    features = extract_hrv_features(ecg_signal)

    if features is None:
        return render_template("index.html")

    X = np.array(features).reshape(1, -1)

    svm_pred = svm.predict(X)[0]
    rf_pred = rf.predict(X)[0]

    final = "CHF" if (svm_pred + rf_pred) >= 1 else "Healthy"

    result = {
        "svm": "CHF" if svm_pred else "Healthy",
        "rf": "CHF" if rf_pred else "Healthy",
        "final": final
    }

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
