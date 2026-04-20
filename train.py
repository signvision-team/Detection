import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib

# Load CSV
df = pd.read_csv("landmarks.csv")

X = df.drop("label", axis=1)
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True
)

# Train SVM
model = SVC(kernel="rbf", probability=True)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

print("\n✔ Model trained on 42 landmark features")
print("✔ Classes:", sorted(y.unique()))
print("✔ Saved as model.pkl")
print(f"✔ Test Accuracy: {model.score(X_test, y_test)*100:.2f}%")