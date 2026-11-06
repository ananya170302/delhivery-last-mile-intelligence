
# ==========================================
# PART D — MODEL PACKAGING
# ==========================================

# Imports
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.shipment_prep import (
    clean_shipments,
    build_features,
    FEATURES
)


# ==========================================
# TASK 1 — LOAD CONFIGURATION AND DATA
# ==========================================

# Create required directories if they do not exist
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Load final model configuration from Part C
config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]

# Extract model configuration
selector_k = int(config["selector_k"])
n_estimators = int(config["n_estimators"])
min_samples_leaf = int(config["min_samples_leaf"])
threshold = float(config["threshold"])

# Convert max_depth configuration
if str(config["max_depth"]).lower() in ["unlimited", "none", "nan"]:
    max_depth = None
else:
    max_depth = int(config["max_depth"])

# Load historical shipment data
raw_data = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# Clean historical data and calculate imputation medians
clean_data, medians = clean_shipments(raw_data)


# Build historical features
feature_data = build_features(clean_data, hubs)

# Create classification target
feature_data["is_delayed"] = (
    feature_data["delivery_hours"] >
    feature_data["promised_hours"]
).astype(int)

# Arrange features in the pinned order
X = feature_data[FEATURES]
y = feature_data["is_delayed"]


print("\nTASK 1 — CONFIGURATION AND DATA")
print("-" * 50)

print("Selector K:", selector_k)
print("Number of estimators:", n_estimators)
print("Max depth:", max_depth)
print("Minimum samples per leaf:", min_samples_leaf)
print("Alert threshold:", threshold)
print("Historical rows:", len(feature_data))


# ==========================================
# TASK 2 — REBUILD STRATIFIED DATA SPLIT
# ==========================================

# First split: 70% training, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Second split: divide temporary data equally
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

print("\nTASK 2 — STRATIFIED DATA SPLIT")
print("-" * 50)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))


# ==========================================
# TASK 3 — REBUILD AND TRAIN FINAL PIPELINE
# ==========================================

# Create the final classification pipeline
final_pipeline = Pipeline([
    (
        "selector",
        SelectKBest(
            score_func=f_classif,
            k=selector_k
        )
    ),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
    )
])

# Train only on the training dataset
final_pipeline.fit(X_train, y_train)

print("\nTASK 3 — FINAL MODEL TRAINING")
print("-" * 50)

print("Final pipeline trained successfully.")


# ==========================================
# TASK 4 — SAVE MODEL ARTIFACTS
# ==========================================

# Save trained model pipeline
joblib.dump(
    final_pipeline,
    "models/delay_model.pkl"
)

# Save alert threshold
pd.DataFrame({
    "threshold": [threshold]
}).to_csv(
    "models/delay_threshold.csv",
    index=False
)

# Save feature order
pd.DataFrame({
    "feature": FEATURES
}).to_csv(
    "models/feature_order.csv",
    index=False
)

# Save historical imputation medians
impute_values = pd.DataFrame([
    {"column": column, "median": median}
    for column, median in medians.items()
])

impute_values.to_csv(
    "models/impute_values.csv",
    index=False
)

print("\nTASK 4 — MODEL ARTIFACTS SAVED")
print("-" * 50)

print("models/delay_model.pkl")
print("models/delay_threshold.csv")
print("models/feature_order.csv")
print("models/impute_values.csv")


# ==========================================
# TASK 5 — RELOAD MODEL AND VERIFY PREDICTIONS
# ==========================================

# Reload the saved model
loaded_pipeline = joblib.load(
    "models/delay_model.pkl"
)

# Generate probabilities using the in-memory pipeline
original_probabilities = final_pipeline.predict_proba(
    X_test
)[:, 1]

# Generate probabilities using the reloaded pipeline
reloaded_probabilities = loaded_pipeline.predict_proba(
    X_test
)[:, 1]

# Calculate maximum absolute probability difference
max_probability_difference = np.max(
    np.abs(
        original_probabilities - reloaded_probabilities
    )
)

# Save reload verification results
reload_check = pd.DataFrame([
    {
        "metric": "test_rows_scored",
        "value": len(X_test)
    },
    {
        "metric": "reload_max_abs_probability_diff",
        "value": max_probability_difference
    },
    {
        "metric": "threshold",
        "value": threshold
    }
])

reload_check.to_csv(
    "outputs/part_d_reload_check.csv",
    index=False
)

print("\nTASK 5 — MODEL RELOAD VERIFICATION")
print("-" * 50)

print("Test rows scored:", len(X_test))
print(
    "Maximum probability difference:",
    max_probability_difference
)
print("Alert threshold:", threshold)

print("\nModel packaging completed successfully!")
