# =========================================================
# URBANPULSE AI - SMART CITY MODEL TRAINING SYSTEM
# =========================================================

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score

print("=" * 60)
print("URBANPULSE AI - MODEL TRAINING")
print("=" * 60)

# =========================================================
# LOAD DATASET
# =========================================================

dataset = pd.read_csv("smart_city_citizen_activity.csv")

print("\nDataset Loaded Successfully!")
print("Rows :", dataset.shape[0])
print("Columns :", dataset.shape[1])

# =========================================================
# ENCODE CATEGORICAL DATA
# =========================================================

gender_encoder = LabelEncoder()
transport_encoder = LabelEncoder()

dataset["Gender_Code"] = gender_encoder.fit_transform(dataset["Gender"])
dataset["Transport_Code"] = transport_encoder.fit_transform(
    dataset["Mode_of_Transport"]
)

# =========================================================
# CREATE ACTIVITY LEVEL
# =========================================================

dataset["Activity_Level"] = pd.cut(
    dataset["Steps_Walked"],
    bins=[0, 5000, 10000, 15000, 25000],
    labels=["Low", "Medium", "High", "Very High"]
)

activity_encoder = LabelEncoder()

dataset["Activity_Code"] = activity_encoder.fit_transform(
    dataset["Activity_Level"].astype(str)
)

# =========================================================
# FEATURE SELECTION
# =========================================================

features = [
    "Age",
    "Gender_Code",
    "Transport_Code",
    "Work_Hours",
    "Shopping_Hours",
    "Entertainment_Hours",
    "Home_Energy_Consumption_kWh",
    "Charging_Station_Usage",
    "Steps_Walked",
    "Calories_Burned",
    "Sleep_Hours",
    "Social_Media_Hours",
    "Public_Events_Hours"
]

X = dataset[features]

# =========================================================
# TARGET 1 - CARBON FOOTPRINT
# =========================================================

y_carbon = dataset["Carbon_Footprint_kgCO2"]

# =========================================================
# TARGET 2 - ACTIVITY LEVEL
# =========================================================

y_activity = dataset["Activity_Code"]

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_carbon,
    test_size=0.2,
    random_state=42
)

# =========================================================
# FEATURE SCALING
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# CARBON FOOTPRINT MODEL
# =========================================================

print("\nTraining Carbon Prediction Model...")

carbon_model = RandomForestRegressor(
    n_estimators=150,
    random_state=42
)

carbon_model.fit(X_train_scaled, y_train)

carbon_predictions = carbon_model.predict(X_test_scaled)

carbon_mae = mean_absolute_error(y_test, carbon_predictions)

print("Carbon Model MAE :", round(carbon_mae, 2))

# =========================================================
# ACTIVITY CLASSIFICATION MODEL
# =========================================================

print("\nTraining Activity Classification Model...")

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X,
    y_activity,
    test_size=0.2,
    random_state=42
)

X_train2_scaled = scaler.fit_transform(X_train2)
X_test2_scaled = scaler.transform(X_test2)

activity_model = RandomForestClassifier(
    n_estimators=120,
    random_state=42
)

activity_model.fit(X_train2_scaled, y_train2)

activity_predictions = activity_model.predict(X_test2_scaled)

activity_accuracy = accuracy_score(y_test2, activity_predictions)

print("Activity Model Accuracy :", round(activity_accuracy * 100, 2), "%")

# =========================================================
# SAVE MODELS
# =========================================================

print("\nSaving Models...")

with open("carbon_model.pkl", "wb") as file:
    pickle.dump(carbon_model, file)

with open("activity_model.pkl", "wb") as file:
    pickle.dump(activity_model, file)

with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

with open("encoders.pkl", "wb") as file:
    pickle.dump({
        "gender": gender_encoder,
        "transport": transport_encoder,
        "activity": activity_encoder
    }, file)

# =========================================================
# SAVE DATA STATISTICS
# =========================================================

statistics = {
    "average_carbon": float(dataset["Carbon_Footprint_kgCO2"].mean()),
    "average_steps": float(dataset["Steps_Walked"].mean()),
    "average_sleep": float(dataset["Sleep_Hours"].mean()),
    "average_energy": float(dataset["Home_Energy_Consumption_kWh"].mean())
}

with open("statistics.pkl", "wb") as file:
    pickle.dump(statistics, file)

print("\nAll Files Saved Successfully!")

print("\nGenerated Files:")
print("1. carbon_model.pkl")
print("2. activity_model.pkl")
print("3. scaler.pkl")
print("4. encoders.pkl")
print("5. statistics.pkl")

print("\nTRAINING COMPLETED")
print("=" * 60)