# model_trainer.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

def train_model(data_path, model_path, scaler_path):
    # Load dataset
    df = pd.read_csv(data_path)

    # Drop rows with missing values
    df = df.dropna(subset=["mean_hr", "std_hr", "min_hr", "max_hr", "range_hr", "label"])
    df["label"] = df["label"].str.strip()

    print("Labels in dataset:", df["label"].unique())

    # Features and target
    X = df[["mean_hr", "std_hr", "min_hr", "max_hr", "range_hr"]]
    y = df["label"]

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into training/testing
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train logistic regression
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
    print("🧮 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"✅ Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    # Save model and scaler
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\n📦 Model saved to {model_path}")
    print(f"📐 Scaler saved to {scaler_path}")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(base_dir, "data", "cardio_dataset_enhanced.csv")
    model_path = os.path.join(base_dir, "ml", "logistic_model.pkl")
    scaler_path = os.path.join(base_dir, "ml", "scaler.pkl")

    train_model(data_path, model_path, scaler_path)
