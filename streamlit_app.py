import streamlit as st
import numpy as np
import os
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from utils import extract_hrv_features

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="CHF Detection using ECG",
    layout="centered"
)

st.title("Early Detection of CHF using ECG")
st.write("Minimal prototype for M.Tech Thesis Part-I")

# -----------------------------
# MODEL HANDLING
# -----------------------------
MODEL_DIR = "model"
SVM_PATH = os.path.join(MODEL_DIR, "svm.pkl")
RF_PATH = os.path.join(MODEL_DIR, "rf.pkl")

def create_demo_models():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X = np.random.rand(60, 5)
    y = np.random.randint(0, 2, 60)

    svm = SVC(kernel="rbf")
    rf = RandomForestClassifier(n_estimators=50)

    svm.fit(X, y)
    rf.fit(X, y)

    joblib.dump(svm, SVM_PATH)
    joblib.dump(rf, RF_PATH)

    return svm, rf

if not os.path.exists(SVM_PATH) or not os.path.exists(RF_PATH):
    svm, rf = create_demo_models()
else:
    svm = joblib.load(SVM_PATH)
    rf = joblib.load(RF_PATH)

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload ECG signal file (.csv or .txt)",
    type=["csv", "txt"]
)

if uploaded_file is not None:
    try:
        ecg_signal = np.loadtxt(uploaded_file)

        st.success("ECG signal loaded successfully")

        features = extract_hrv_features(ecg_signal)

        if features is None:
            st.error("ECG signal too short for HRV extraction")
        else:
            X = np.array(features).reshape(1, -1)

            svm_pred = svm.predict(X)[0]
            rf_pred = rf.predict(X)[0]

            final = "CHF" if (svm_pred + rf_pred) >= 1 else "Healthy"

            st.subheader("Prediction Results")
            st.write(f"**SVM Prediction:** {'CHF' if svm_pred else 'Healthy'}")
            st.write(f"**Random Forest Prediction:** {'CHF' if rf_pred else 'Healthy'}")
            st.write(f"### Final Prediction: {final}")

            st.info(
                "Note: This is a prototype demo. "
                "Final model training and validation will be done in the next semester."
            )

    except Exception as e:
        st.error("Error processing ECG file")

