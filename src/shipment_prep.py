import pandas as pd
import numpy as np

# Promised delivery time based on service type
PROMISED_HOURS = {
    "Express": 36,
    "Standard": 60,
    "Economy": 96
}

# Features used by the prediction models
FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no"
]


# Columns representing operational noise
NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no"
]

# Clean shipment data
def clean_shipments(raw, medians=None):
    
    clean_df = raw.copy()        # Work on a copy to preserve the original dataset

    # 0. Capture original data quality metrics
    print("\n--- Before Cleaning ---")

    print("Total rows:", len(clean_df))
    print("Total columns:", len(clean_df.columns))
    print("Duplicate rows:", clean_df.duplicated().sum())

    print("\nMissing values:")
    print(clean_df.isnull().sum())

    print("\nTotal missing values:", clean_df.isnull().sum().sum())

    # 1. Remove exact duplicate records
    print("Total duplicates found in raw dataset:", clean_df.duplicated().sum())
    print("Percent of duplicates:", round((clean_df.duplicated().sum())/len(clean_df)*100, 2), "%")   # 1.96 % which is very small so we drop them
    clean_df = clean_df.drop_duplicates()
    print("Total duplicates found in raw dataset after dropping initial set of duplicates:", clean_df.duplicated().sum())

    # 2. Standardize service type
    print("\n", clean_df.head())
    clean_df["service_type"] = (clean_df["service_type"].str.strip().str.title())
    print("\n", clean_df.head())

    # 3. Handle missing payment modes
    print("\nTotal missing values in payment_mode:", clean_df["payment_mode"].isnull().sum())
    clean_df["payment_mode"] = clean_df["payment_mode"].fillna("Unknown")
    print("Total missing values in payment_mode after filling them:", clean_df["payment_mode"].isnull().sum())

    # 4. Convert declared value into numeric INR
    print("\nRaw declared value column values:\n",clean_df["declared_value"].head())
    clean_df["declared_value_inr"] = (clean_df["declared_value"]
                                        .str.replace("₹", "", regex=False)
                                        .str.replace(",", "", regex=False))

    clean_df["declared_value_inr"] = pd.to_numeric(
        clean_df["declared_value_inr"]
    )

    print("\nCleaned declared value column values:\n",clean_df["declared_value_inr"].head())

    clean_df = clean_df.drop(columns=["declared_value"])

    # 5. Convert dispatch time into datetime
    print("\nRaw dispatch time column values:\n",clean_df["dispatch_time"].head())
    clean_df["dispatch_time"] = pd.to_datetime(clean_df["dispatch_time"])
    print("\nCleaned dispatch time column values:\n",clean_df["dispatch_time"].head())

    # 6. Handle invalid parcel weights
    print("\nRaw parcel weights column values:\n",clean_df["parcel_weight_kg"].head())
    print("\nTotal invlaid parcel weights before cleaning:", (clean_df["parcel_weight_kg"]>100).sum())
    # Identify rows where parcel weight is greater than 100 kg.
    # The condition returns True for invalid weights and False for valid weights.
    clean_df.loc[
    clean_df["parcel_weight_kg"] > 100,  # Select rows with weight above 100 kg
    "parcel_weight_kg"                   # Select only the parcel_weight_kg column
    ] = np.nan                           # Replace the selected values with NaN (missing values)

    print("\nCleaned parcel weights column values:\n",clean_df["parcel_weight_kg"].head())
    print("\nTotal invlaid parcel weights after cleaning:", (clean_df["parcel_weight_kg"]>100).sum())


    # 7. Calculate medians from historical data if not provided
    if medians is None:
        medians = {
            "distance_km": float(round(clean_df["distance_km"].median(), 2)),
            "parcel_weight_kg": float(round(clean_df["parcel_weight_kg"].median(), 2)),
            "declared_value_inr": float(round(clean_df["declared_value_inr"].median(), 2)),
        }

    # Fill missing numerical values using medians
    print("\nTotal missing values in distance_km column:", clean_df["distance_km"].isnull().sum())
    print("Median for column distance_km:", medians["distance_km"])
    clean_df["distance_km"] = clean_df["distance_km"].fillna(medians["distance_km"])
    print("Total missing values in distance_km column after cleaning:", clean_df["distance_km"].isnull().sum())

    print("\nTotal missing values in parcel_weight_kg column:", clean_df["parcel_weight_kg"].isnull().sum())
    print("Median for column parcel_weight_kg:", medians["parcel_weight_kg"])
    clean_df["parcel_weight_kg"] = clean_df["parcel_weight_kg"].fillna(medians["parcel_weight_kg"])
    print("Total missing values in parcel_weight_kg column after cleaning:", clean_df["parcel_weight_kg"].isnull().sum())

    print("\nTotal missing values in declared_value_inr column:", clean_df["declared_value_inr"].isnull().sum())
    print("Median for column declared_value_inr:", medians["declared_value_inr"])
    clean_df["declared_value_inr"] = clean_df["declared_value_inr"].fillna(medians["declared_value_inr"])
    print("Total missing values in declared_value_inr column after cleaning:", clean_df["declared_value_inr"].isnull().sum())

    return clean_df, medians

# Build features and enrich shipment data
def build_features(clean, hubs):
    features_df = clean.copy()
    print("\nOriginal features_df:\n", features_df)

    # 1. Attach hub information using a left join on hub_id
    features_df = clean.merge(
        hubs[
            ["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]
        ],
        on = "hub_id",
        how = "left"
    )
    print("\nfeatures_df after left join:\n", features_df)


    # 2. Map service type to its promised delivery time in hours
    features_df["promised_hours"] = (
        features_df["service_type"].map(PROMISED_HOURS)
    )
    # print("\nfeatures_df after mapping service type to its promised delivery time in hours:\n", features_df)

    # 3. Identify Cash on Delivery shipments - 1 when payment_mode is COD, else 0 — Unknown gives 0
    features_df["is_cod"] = (features_df["payment_mode"] == "COD").astype(int)

    # 4. Identify weekend dispatches - 1 when the dispatch day is Saturday or Sunday
    features_df["is_weekend"] = (features_df["dispatch_time"].dt.dayofweek.isin([5, 6])).astype(int)

    # 5. Extract the dispatch hour as an integer
    features_df["dispatch_hour"] = (features_df["dispatch_time"].dt.hour)

    # 6. Apply natural logarithm to declared value
    features_df["log_declared_value"] = np.log(features_df["declared_value_inr"])

    print("\nFinal features_df:\n", features_df)

    return features_df
