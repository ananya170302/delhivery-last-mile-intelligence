# """
# Part A: Data Foundation and Delay Evidence

# Source datasets:
#     1. data/raw_shipments.csv
#     2. data/hubs.csv
#     3. data/new_shipments.csv
# """


import pandas as pd

from src.shipment_prep import clean_shipments, build_features, FEATURES


def main():

    # 1. Load historical shipment data and hub master
    raw = pd.read_csv("data/raw_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")

    print("Raw shipments shape:", raw.shape)

    # 2. Remove duplicates separately for before-cleaning checks
    deduplicated_df = raw.drop_duplicates().copy()

    duplicate_rows_removed = len(raw) - len(deduplicated_df)

    # 3. Capture metrics before cleaning and imputation
    missing_distance_km_before = deduplicated_df["distance_km"].isna().sum()

    # Count missing weights separately from invalid sentinel values
    missing_parcel_weight_before = deduplicated_df["parcel_weight_kg"].isna().sum()

    sentinel_weight_rows = (
        deduplicated_df["parcel_weight_kg"] > 100
    ).sum()

    missing_declared_value_before = deduplicated_df["declared_value"].isna().sum()
    missing_payment_mode_before = deduplicated_df["payment_mode"].isna().sum()

    service_type_spellings_before = deduplicated_df["service_type"].nunique()

    # 4. Clean historical shipment data
    clean_df, medians = clean_shipments(raw)

    service_type_spellings_after = clean_df["service_type"].nunique()

    print("Cleaned shipments shape:", clean_df.shape)
    print("Historical medians:", medians)

    # 5. Build features
    features_df = build_features(clean_df, hubs)

    rows_after_hub_merge = len(features_df)

    # 6. Create target variable
    features_df["is_delayed"] = (
        features_df["delivery_hours"] > features_df["promised_hours"]
    ).astype(int)

    # 7. Save modelling table
    features_df.to_csv("data/modelling_table.csv", index=False)

    # 8. Calculate declared-value metrics using median-filled data
    declared_value_mean = features_df["declared_value_inr"].mean()
    declared_value_median = features_df["declared_value_inr"].median()
    log_declared_value_mean = features_df["log_declared_value"].mean()
    log_declared_value_median = features_df["log_declared_value"].median()

    # 9. Calculate delivery-hour outlier metrics
    q1 = features_df["delivery_hours"].quantile(0.25)
    q3 = features_df["delivery_hours"].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    outlier_count = (
        (features_df["delivery_hours"] < lower_bound)
        | (features_df["delivery_hours"] > upper_bound)
    ).sum()

    # 10. Count missing cells across all features and target columns
    validation_columns = FEATURES + ["delivery_hours", "is_delayed"]

    nulls_in_features_and_target = (
        features_df[validation_columns].isna().sum().sum()
    )

    # 11. Prepare checks output
    checks = {
        "raw_rows": len(raw),
        "duplicate_rows_removed": duplicate_rows_removed,
        "rows_after_dedup": len(deduplicated_df),
        "missing_distance_km_before": missing_distance_km_before,
        "missing_parcel_weight_before": missing_parcel_weight_before,
        "missing_declared_value_before": missing_declared_value_before,
        "missing_payment_mode_before": missing_payment_mode_before,
        "sentinel_weight_rows": sentinel_weight_rows,
        "median_distance_km": medians["distance_km"],
        "median_parcel_weight_kg": medians["parcel_weight_kg"],
        "median_declared_value_inr": medians["declared_value_inr"],
        "service_type_spellings_before": service_type_spellings_before,
        "service_type_spellings_after": service_type_spellings_after,
        "rows_after_hub_merge": rows_after_hub_merge,
        "nulls_in_features_and_target": nulls_in_features_and_target,
        "delayed_count": int(features_df["is_delayed"].sum()),
        "delay_rate": features_df["is_delayed"].mean(),
        "declared_value_mean_inr": declared_value_mean,
        "declared_value_median_inr": declared_value_median,
        "log_declared_value_mean": log_declared_value_mean,
        "log_declared_value_median": log_declared_value_median,
        "hours_q1": q1,
        "hours_q3": q3,
        "hours_iqr": iqr,
        "hours_lower_bound": lower_bound,
        "hours_upper_bound": upper_bound,
        "hours_outlier_count": outlier_count
    }

    # 12. Write checks CSV
    checks_df = pd.DataFrame(
        list(checks.items()),
        columns=["metric", "value"]
    )

    checks_df["value"] = checks_df["value"].round(4)

    checks_df.to_csv("outputs/part_a_checks.csv", index=False)

    # 13. Generate delay evidence by group

    # Create distance bands
    features_df["distance_band"] = pd.cut(
        features_df["distance_km"],
        bins=[0, 200, 400, 600, 2000],
        labels=["0-200", "201-400", "401-600", "601+"],
        right=True,
        include_lowest=True
    )

    # Dimensions to analyse
    dimensions = ["service_type", "hub_tier", "distance_band"]

    delay_results = []

    for dimension in dimensions:

        grouped = (
            features_df.groupby(dimension, observed=True)
            .agg(
                shipments=("shipment_id", "count"),
                delayed=("is_delayed", "sum"),
                delay_rate=("is_delayed", "mean"),
                mean_hours=("delivery_hours", "mean")
            )
            .reset_index()
        )

        # Rename grouping column to a common name
        grouped = grouped.rename(columns={dimension: "group"})

        # Add dimension name
        grouped.insert(0, "dimension", dimension)

        delay_results.append(grouped)

    # Combine all three dimensions
    delay_by_group_df = pd.concat(
        delay_results,
        ignore_index=True
    )

    # Round calculated metrics
    delay_by_group_df["delay_rate"] = delay_by_group_df["delay_rate"].round(4)
    delay_by_group_df["mean_hours"] = delay_by_group_df["mean_hours"].round(4)

    # Arrange columns in the required order
    delay_by_group_df = delay_by_group_df[
        ["dimension", "group", "shipments", "delayed", "delay_rate", "mean_hours"]
    ]

    # Save delay evidence CSV
    delay_by_group_df.to_csv(
        "outputs/part_a_delay_by_group.csv",
        index=False
    )

    print("\\n--- Delay Evidence by Group ---")
    print(delay_by_group_df.to_string(index=False))

    # 14. Generate correlation matrix

    # Select all numeric columns from the modelling table
    numeric_df = features_df.select_dtypes(include="number")

    # Calculate Pearson correlation matrix
    correlation_df = numeric_df.corr()

    # Save correlation matrix with row labels as the first column
    correlation_df.to_csv(
        "outputs/part_a_correlations.csv",
        index=True,
        index_label="column"
    )

    print("\n--- Correlation Matrix ---")
    print(correlation_df.round(4).to_string())

    # 15. Final validation
    print("\n--- Part A Checks ---")
    print(checks_df.to_string(index=False))

    print("\n--- Modelling Table Validation ---")
    print("Final shape:", features_df.shape)
    print("Total shipments:", len(features_df))
    print("Delayed shipments:", features_df["is_delayed"].sum())
    print("Delay rate:", round(features_df["is_delayed"].mean(), 4))
    print("Missing values:", features_df.isnull().sum().sum())

    print("\nModelling table and checks file saved successfully!")

if __name__ == "__main__":
    main()




# Initial exploration of historical shipments, new shipments and hub master data.
'''
# def load_data():
#     raw = pd.read_csv("data/raw_shipments.csv")
#     hubs = pd.read_csv("data/hubs.csv")
#     new_shipments = pd.read_csv("data/new_shipments.csv")

#     return raw, hubs, new_shipments

# def analyze_data(df, dataset_name):
#     print("\n" + "-" * 60)
#     print(f"{dataset_name.upper()} - Data Profile")
#     print("-" * 60)

#     # 1. Dataset dimensions
#     print(f"\nRows: {df.shape[0]}")
#     print(f"Columns: {df.shape[1]}")

#     # 2. Column names and data types
#     print("\nColumn names and data types:")
#     print(df.dtypes)

#     # 3. Missing values
#     print("\nMissing values:")
#     missing = df.isnull().sum()

#     if missing.any():
#         print(missing[missing > 0])
#     else:
#         print("No missing values")

#     # 4. Duplicate records
#     print(f"\nDuplicate rows: {df.duplicated().sum()}")

#     # 5. Numerical summary
#     print("\nNumerical summary:")
#     print(df.describe().round(2))

# def main():
#     raw, hubs, new_shipments = load_data()   # load all datasets

#     analyze_data(raw, "Raw Shipments")
#     analyze_data(hubs, "Hub Master")
#     analyze_data(new_shipments, "New Shipments")
'''




# Testing for shipment_prep functions: clean_shipments() on raw
'''
# from src.shipment_prep import clean_shipments
# raw = pd.read_csv("data/raw_shipments.csv")
# # clean_shipments(raw)

# clean_df, medians = clean_shipments(raw)

# print("\nCleaned dataset shape:", clean_df.shape)
# print("\nCalculated medians:", medians)
# print("\nMissing values after cleaning:")
# print(clean_df.isnull().sum())
'''



# Testing for shipment_prep functions: build_features() on raw
'''
# from src.shipment_prep import build_features

# hubs = pd.read_csv("data/hubs.csv")

# features_df = build_features(clean_df, hubs)

# print("\nFeature dataset shape:", features_df.shape)
# print("\nHub columns:")
# print(features_df[["hub_id", "hub_city", "region", "hub_tier"]].head())

# print("\nShipment rows:", len(clean_df))
# print("Rows after hub merge:", len(features_df))
'''







# Testing for shipment_prep functions: clean_shipments() and build_features() on new_shipments
'''
from src.shipment_prep import clean_shipments, build_features, FEATURES

# Load datasets
raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
new_shipments = pd.read_csv("data/new_shipments.csv")

# Calculate medians using historical data
clean_df, medians = clean_shipments(raw)

# Clean new shipments using the historical medians
new_clean, _ = clean_shipments(new_shipments, medians=medians)

# Build features for new shipments
new_features = build_features(new_clean, hubs)

# Validate the results
print("\n--- New Shipments Validation ---")

print("New shipments shape:", new_features.shape)
print("Has delivery_hours:", "delivery_hours" in new_features.columns)

print(
    "Missing required feature values:",
    new_features[FEATURES].isnull().sum().sum()
)

print("\nMissing hub information:")
print(
    new_features[
        ["hub_city", "region", "hub_tier", "hub_daily_capacity"]
    ].isnull().sum()
)
'''