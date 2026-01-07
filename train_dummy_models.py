import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

os.makedirs("model", exist_ok=True)

X = np.random.rand(60, 5)
y = np.random.randint(0, 2, 60)

svm = SVC(kernel='rbf')
rf = RandomForestClassifier(n_estimators=50)

svm.fit(X, y)
rf.fit(X, y)

joblib.dump(svm, "model/svm.pkl")
joblib.dump(rf, "model/rf.pkl")

print("Demo models saved.")
