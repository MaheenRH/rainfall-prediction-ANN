import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

# -----------------------------
# 📦 Load the saved model
# -----------------------------
model_path = 'mymodel.h5'  # or 'mymodel.h5' if you saved in HDF5
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found. Train the model first.")

print(f"✅ Loading trained model from {model_path}...")
model = load_model(model_path)

# -----------------------------
# 🌦️ Load and preprocess dataset (same steps as training)
# -----------------------------
dataset = pd.read_csv('data/austin_weather.csv')

columns_to_clean = [
    'TempHighF', 'TempAvgF', 'TempLowF', 'DewPointHighF', 'DewPointAvgF',
    'DewPointLowF', 'HumidityHighPercent', 'HumidityAvgPercent', 'HumidityLowPercent',
    'SeaLevelPressureHighInches', 'SeaLevelPressureAvgInches', 'SeaLevelPressureLowInches',
    'VisibilityHighMiles', 'VisibilityAvgMiles', 'VisibilityLowMiles',
    'WindHighMPH', 'WindAvgMPH', 'WindGustMPH', 'PrecipitationSumInches'
]

for col in columns_to_clean:
    dataset[col] = pd.to_numeric(dataset[col], errors='coerce')
    dataset[col].fillna(dataset[col].mean(), inplace=True)

if 'Date' in dataset.columns:
    dataset.drop(columns=['Date'], inplace=True)

y = dataset['Events'].fillna('No Rain').values
X = dataset.drop(columns=['Events'], errors='ignore').values

# Encode target labels
lb = LabelEncoder()
y = lb.fit_transform(y)

# Scale features
sc = StandardScaler()
X = sc.fit_transform(X)

# -----------------------------
# 🔍 Evaluate the model
# -----------------------------
print("🔎 Generating predictions...")
y_pred = np.argmax(model.predict(X), axis=1)

# Confusion matrix
print("\n📊 Classification Report:")
print(classification_report(y, y_pred))

cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lb.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Rainfall Prediction ANN")
plt.tight_layout()

# Create visuals directory if missing
os.makedirs("visuals", exist_ok=True)
plt.savefig("visuals/confusion_matrix.png", bbox_inches='tight')
plt.show()

print("✅ Evaluation complete. Confusion matrix saved in 'visuals/confusion_matrix.png'.")

plt.figure()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['loss'], label='loss')
plt.title("Training Accuracy & Loss")
plt.legend()
plt.savefig("visuals/training_curve.png")
