import pandas as pd
import numpy as np
import joblib
from sklearn.svm             import SVC
from sklearn.ensemble        import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics         import classification_report, accuracy_score

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
print("Loading landmarks.csv …")
df = pd.read_csv("landmarks.csv")

X = df.drop("label", axis=1).values
y = df["label"].values

print(f"  Samples  : {len(X)}")
print(f"  Features : {X.shape[1]}")
print(f"  Classes  : {sorted(set(y))}")

# ─────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ─────────────────────────────────────────────
# BUILD ENSEMBLE MODEL
#
#   SVM  — good at high-dimensional boundaries
#   RF   — robust, handles noise well
#   Soft voting averages their probability outputs
# ─────────────────────────────────────────────
svm_clf = SVC(kernel="rbf", C=10, gamma="scale",
              probability=True, random_state=42)

rf_clf  = RandomForestClassifier(
              n_estimators=200,
              max_depth=None,
              min_samples_split=2,
              random_state=42,
              n_jobs=-1
          )

ensemble = VotingClassifier(
    estimators=[("svm", svm_clf), ("rf", rf_clf)],
    voting="soft"          # averages predict_proba → confidence scores work
)

# Wrap in a pipeline so scaling is saved with the model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    ensemble)
])

# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
print("\nTraining ensemble (SVM + Random Forest) …  (may take 1–3 mins)")
model.fit(X_train, y_train)

# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print(f"\n{'─'*50}")
print(f"  Test Accuracy : {acc*100:.2f}%")
print(f"{'─'*50}")
print("\nPer-class report:")
print(classification_report(y_test, y_pred))

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
joblib.dump(model, "model.pkl")
print("  ✔ Model saved → model.pkl")
print("  ✔ Includes StandardScaler + SVM + RandomForest ensemble")
print("  ✔ Trained on 63 normalised x,y,z features")