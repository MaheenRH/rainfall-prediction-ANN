import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# -----------------------------
# 🌧️ Rainfall Prediction using ANN
# -----------------------------

print("🔹 Loading dataset...")
dataset = pd.read_csv('data/austin_weather.csv')

print(f"✅ Dataset loaded successfully with {dataset.shape[0]} rows and {dataset.shape[1]} columns.")

# -----------------------------
# 🧹 Data Cleaning
# -----------------------------
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

# Drop non-numeric or irrelevant columns like 'Date' if present
if 'Date' in dataset.columns:
    dataset.drop(columns=['Date'], inplace=True)

# -----------------------------
# 🧠 Feature Engineering
# -----------------------------
# Encode target variable
y = dataset['Events'].fillna('No Rain').values
X = dataset.drop(columns=['Events'], errors='ignore').values

# Encode categorical target values
lb = LabelEncoder()
y = lb.fit_transform(y)

# Scale numerical features
sc = StandardScaler()
X = sc.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"📊 Data split complete: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples.")

# -----------------------------
# 🧩 Build ANN Model
# -----------------------------
num_features = X_train.shape[1]
num_classes = len(np.unique(y))

model = Sequential()
model.add(Dense(units=21, activation='relu', input_shape=(num_features,)))
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("🚀 Training model (this may take a few minutes)...")
history = model.fit(X_train, y_train, epochs=1000, batch_size=128, verbose=0)

# -----------------------------
# 📈 Evaluate Model
# -----------------------------
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Model training complete.")
print(f"🎯 Test Accuracy: {accuracy * 100:.2f}%")

# -----------------------------
# 💾 Save Model
# -----------------------------
model.save('mymodel.h5')
print("💾 Model saved as 'mymodel.h5' successfully.")

