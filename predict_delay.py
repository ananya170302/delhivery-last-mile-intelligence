
# ==========================================
# PART D — SCORE NEW SHIPMENTS
# ==========================================

# ==========================================
# Imports
# ==========================================

import pandas as pd
import joblib

from src.shipment_prep import (
    clean_shipments,
    build_features
)


# ==========================================
# TASK 1 — LOAD INPUT DATA AND MODEL ARTIFACTS
# ==========================================

# Load new shipments and hub reference data
new_shipments = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# Load packaged model
model = joblib.load("models/delay_model.pkl")

# Load alert threshold
threshold_data = pd.read_csv("models/delay_threshold.csv")
threshold = float(threshold_data.loc[0, "threshold"])

# Load feature order
feature_order_data = pd.read_csv("models/feature_order.csv")
feature_order = feature_order_data["feature"].tolist()

# Load historical imputation values
impute_data = pd.read_csv("models/impute_values.csv")

stored_medians = dict(
    zip(
        impute_data["column"],
        impute_data["median"]
    )
)

# Load hub segments
hub_segments = pd.read_csv("outputs/hub_segments.csv")

print("\nTASK 1 — INPUT DATA AND MODEL LOADED")
print("-" * 50)

print("New shipments:", len(new_shipments))
print("Hub records:", len(hubs))
print("Model loaded successfully.")
print("Alert threshold:", threshold)
print("Number of model features:", len(feature_order))


# ==========================================
# TASK 2 — CLEAN NEW SHIPMENTS
# ==========================================

# Use the same cleaning function as Part A.
# Pass historical medians to avoid calculating
# new imputation values from the incoming data.

clean_data, _ = clean_shipments(
    new_shipments,
    medians=stored_medians
)

print("\nTASK 2 — NEW SHIPMENTS CLEANED")
print("-" * 50)

print("Rows after cleaning:", len(clean_data))


# ==========================================
# TASK 3 — BUILD MODEL FEATURES
# ==========================================

# Attach hub information and generate features
feature_data = build_features(
    clean_data,
    hubs
)

# Arrange features in the exact order used during training
X_new = feature_data[feature_order]

print("\nTASK 3 — MODEL FEATURES CREATED")
print("-" * 50)

print("Feature rows:", len(X_new))
print("Feature columns:", len(X_new.columns))


# ==========================================
# TASK 4 — PREDICT DELAY PROBABILITIES
# ==========================================

# Get probability of the positive class (delayed = 1)
delay_probabilities = model.predict_proba(X_new)[:, 1]

# Round probabilities to 4 decimal places
feature_data["delay_probability"] = (
    delay_probabilities.round(4)
)

# Flag shipments when probability >= stored threshold
feature_data["flagged"] = (
    feature_data["delay_probability"] >= threshold
).astype(int)

print("\nTASK 4 — DELAY PREDICTIONS GENERATED")
print("-" * 50)

print("Total shipments scored:", len(feature_data))
print("Flagged shipments:", feature_data["flagged"].sum())
print("Unflagged shipments:", (feature_data["flagged"] == 0).sum())


# ==========================================
# TASK 5 — JOIN HUB SEGMENTS
# ==========================================

# Attach the operational segment assigned to each hub
feature_data = feature_data.merge(
    hub_segments[["hub_id", "segment"]],
    on="hub_id",
    how="left",
    sort=False,
    validate="many_to_one"
)

# Verify every shipment received a hub segment
if feature_data["segment"].isna().any():
    raise ValueError("Some shipments could not be matched to a hub segment.")

print("\nTASK 5 — HUB SEGMENTS ATTACHED")
print("-" * 50)

print("Shipments with hub segments:", feature_data["segment"].notna().sum())


# ==========================================
# TASK 6 — ASSIGN OPERATIONAL ACTIONS
# ==========================================

# Assign actions based on the delay flag and hub segment

feature_data["action"] = "No action"

# Flagged shipments from Escalate hubs
escalate_mask = (
    (feature_data["flagged"] == 1) &
    (feature_data["segment"] == "Escalate")
)

feature_data.loc[
    escalate_mask,
    "action"
] = "Escalate to hub manager"

# Flagged shipments from Standard or Watch hubs
customer_call_mask = (
    (feature_data["flagged"] == 1) &
    (feature_data["segment"] != "Escalate")
)

feature_data.loc[
    customer_call_mask,
    "action"
] = "Proactive customer call"

print("\nTASK 6 — OPERATIONAL ACTIONS ASSIGNED")
print("-" * 50)

print(feature_data["action"].value_counts())


# ==========================================
# TASK 7 — SAVE FINAL SHIPMENT ACTIONS
# ==========================================

# Select the exact output columns and preserve input row order
output_columns = [
    "shipment_id",
    "hub_id",
    "segment",
    "delay_probability",
    "flagged",
    "action"
]

final_output = feature_data[output_columns].copy()

# Save output without an index
final_output.to_csv(
    "outputs/new_shipment_actions.csv",
    index=False
)

print("\nTASK 7 — FINAL OUTPUT SAVED")
print("-" * 50)

print("Output file: outputs/new_shipment_actions.csv")
print("Total output rows:", len(final_output))

print("\nAction summary:")
print(
    final_output["action"].value_counts()
)

print("\nFirst 10 shipment predictions:")
print(final_output.head(10))

print("\nNew shipment scoring completed successfully!")
