from flask import Flask, render_template, request
import numpy as np
import joblib
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from utils import extract_hrv_features

app = Flask(__name__)

# ----------------------------
# MODEL PATH HANDLING (IMPORTANT)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
SVM_PATH = os.path.join(MODEL_DIR, "svm.pkl")
RF_PATH = os.path.join(MODEL_DIR, "rf.pkl")

# ----------------------------
# CREATE DEMO MODELS IF MISSING
# ----------------------------
def create_demo_models():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Dummy data (DEMO ONLY)
    X = np.random.rand(60, 5)
    y = np.random.randint(0, 2, 60)

    svm = SVC(kernel="rbf")
    rf = RandomForestClassifier(n_estimators=50)

    svm.fit(X, y)
    rf.fit(X, y)

    joblib.dump(svm, SVM_PATH)
    joblib.dump(rf, RF_PATH)

    return svm, rf

# ----------------------------
# LOAD OR CREATE MODELS
# ----------------------------
if not os.path.exists(SVM_PATH) or not os.path.exists(RF_PATH):
    svm, rf = create_demo_models()
else:
    svm = joblib.load(SVM_PATH)
    rf = joblib.load(RF_PATH)

# ----------------------------
# ROUTES
# ----------------------------
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
