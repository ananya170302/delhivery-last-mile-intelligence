import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from src.shipment_prep import FEATURES

# TASK 1: DATA SPLITTING INTO TRAINING, VALIDATION AND TEST SETS

# 1. Load the modelling table
df = pd.read_csv("data/modelling_table.csv")
# print(df)

# 2. Separate input features (X) and target (y)
X = df[FEATURES]
y = df["delivery_hours"]

# 3. Split 70% for training and 30% for temporary data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Split the temporary 30% equally into validation and test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 5. Display split details
print("PART B - TASK 1: DATA SPLITTING")
print("--------------------------------")

print("Original dataset:", X.shape)
print("Training dataset:", X_train.shape)
print("Validation dataset:", X_val.shape)
print("Test dataset:", X_test.shape)

print("\nTarget shapes:")
print("Training:", y_train.shape)
print("Validation:", y_val.shape)
print("Test:", y_test.shape)



# TASK 2: SCALING WITHOUT DATA LEAKAGE

# 1. Initialize StandardScaler
scaler = StandardScaler()

# 2. Fit scaler using training data only
X_train_scaled = scaler.fit_transform(X_train)

# 3. Transform test data using training statistics
X_test_scaled = scaler.transform(X_test)

# 4. Calculate maximum absolute column mean
train_scaled_max_abs_mean = np.abs(X_train_scaled.mean(axis=0)).max()
test_scaled_max_abs_mean = np.abs(X_test_scaled.mean(axis=0)).max()

# 5. Prepare validation metrics
checks = {
    "train_rows": len(X_train),
    "val_rows": len(X_val),
    "test_rows": len(X_test),
    "train_scaled_max_abs_mean": train_scaled_max_abs_mean,
    "test_scaled_max_abs_mean": test_scaled_max_abs_mean
}

# 6. Save checks to CSV
checks_df = pd.DataFrame(
    list(checks.items()),
    columns=["metric", "value"]
)

checks_df.to_csv(
    "outputs/part_b_checks.csv",
    index=False
)

# 7. Display results
print("\nTASK 2: SCALING VALIDATION")
print("-------------------------")
print(checks_df.to_string(index=False))

print("\nScaled training shape:", X_train_scaled.shape)
print("Scaled test shape:", X_test_scaled.shape)



# TASK 3: MEAN BASELINE REGRESSION

# 1. Calculate average delivery hours from training data
baseline_prediction = y_train.mean()

# 2. Predict the same average for every validation shipment
y_val_baseline_pred = np.full(
    shape=len(y_val),
    fill_value=baseline_prediction
)

# 3. Calculate evaluation metrics
baseline_mae = mean_absolute_error(y_val, y_val_baseline_pred)
baseline_mse = mean_squared_error(y_val, y_val_baseline_pred)
baseline_rmse = np.sqrt(baseline_mse)
baseline_r2 = r2_score(y_val, y_val_baseline_pred)

# 4. Display results
print("\nTASK 3: MEAN BASELINE REGRESSION")
print("--------------------------------")
print(f"Training mean delivery hours: {baseline_prediction:.4f}")
print(f"Validation MAE: {baseline_mae:.4f}")
print(f"Validation MSE: {baseline_mse:.4f}")
print(f"Validation RMSE: {baseline_rmse:.4f}")
print(f"Validation R2: {baseline_r2:.4f}")




# TASK 4: DECISION TREE DEPTH DIAGNOSTIC

# Depths to evaluate
depths = [1, 2, 3, 4, 6, 8, 12, None]

tree_results = []

for depth in depths:
    # Initialize Decision Tree
    tree_model = DecisionTreeRegressor(max_depth=depth, random_state=42)

    # Train using training data
    tree_model.fit(X_train, y_train)

    # Predictions on training data
    train_pred = tree_model.predict(X_train)

    # Predictions on validation data
    val_pred = tree_model.predict(X_val)

    # Calculate RMSE
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    # Store results
    tree_results.append({
        "max_depth": "unlimited" if depth is None else str(depth),
        "train_rmse": train_rmse,
        "val_rmse": val_rmse
    })

# Convert results into DataFrame
tree_depth_df = pd.DataFrame(tree_results)

# Save results
tree_depth_df.to_csv("outputs/part_b_tree_depth.csv", index=False)

# Identify best depth based on validation RMSE
best_tree_row = tree_depth_df.loc[
    tree_depth_df["val_rmse"].idxmin()
]

# Display Results
print("\nTASK 4: DECISION TREE DEPTH DIAGNOSTIC")
print("--------------------------------------")
print(tree_depth_df.to_string(index=False))

print("\nBest tree depth:", best_tree_row["max_depth"])
print("Best validation RMSE:", round(best_tree_row["val_rmse"], 4))





# TASK 5: COMPARING FOUR REGRESSION MODELS

# Store model evaluation results
model_results = []

# Helper function to calculate evaluation metrics
def evaluate_model(model_name, y_actual, y_pred):
    mae = mean_absolute_error(y_actual, y_pred)
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_actual, y_pred)

    model_results.append({
        "model": model_name,
        "split": "val",
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

# 1. Mean Baseline
evaluate_model("mean_baseline", y_val, y_val_baseline_pred)

# 2. Linear Regression
linear_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

linear_model.fit(X_train, y_train)
linear_model_pred = linear_model.predict(X_val)

evaluate_model("linear_regression", y_val, linear_model_pred)

# 3. Decision Tree (best depth from Task 4)
decision_tree_model = DecisionTreeRegressor(
    max_depth=6,
    random_state=42
)

decision_tree_model.fit(X_train, y_train)
tree_val_pred = decision_tree_model.predict(X_val)

evaluate_model(
    "decision_tree",
    y_val,
    tree_val_pred
)

# 4. Random Forest
random_forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

random_forest_model.fit(X_train, y_train)
rf_val_pred = random_forest_model.predict(X_val)

evaluate_model(
    "random_forest",
    y_val,
    rf_val_pred
)

# 5. Gradient Boosting
gradient_boosting_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

gradient_boosting_model.fit(X_train, y_train)
gb_val_pred = gradient_boosting_model.predict(X_val)

evaluate_model(
    "gradient_boosting",
    y_val,
    gb_val_pred
)

# Convert results into DataFrame
metrics_df = pd.DataFrame(model_results)

# Save validation metrics
metrics_df.to_csv("outputs/part_b_metrics.csv", index=False)

# Display results
print("\nTASK 5: REGRESSION MODEL COMPARISON")
print("-----------------------------------")
print(metrics_df.to_string(index=False))


# Identify best model using validation RMSE
best_model_row = metrics_df.loc[metrics_df["RMSE"].idxmin()]

print("\nBest validation model:", best_model_row["model"])
print("Best validation RMSE:", round(best_model_row["RMSE"], 4))




# TASK 6: GRADIENT BOOSTING LEARNING RATE EXPERIMENT

# Learning rates to evaluate
learning_rates = [0.01, 0.1, 1.0]

gb_results = []

for lr in learning_rates:
    # Initialize Gradient Boosting model
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=lr,
        random_state=42
    )

    # Train the model
    gb_model.fit(X_train, y_train)

    # Predict validation delivery hours
    val_pred = gb_model.predict(X_val)

    # Calculate validation metrics
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_r2 = r2_score(y_val, val_pred)

    # Store results
    gb_results.append({
        "learning_rate": lr,
        "val_rmse": val_rmse,
        "val_r2": val_r2
    })

# Convert results into DataFrame
gb_lr_df = pd.DataFrame(gb_results)

# Save results
gb_lr_df.to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

# Identify best learning rate
best_lr_row = gb_lr_df.loc[gb_lr_df["val_rmse"].idxmin()]

print("\nTASK 6: GRADIENT BOOSTING LEARNING RATE")
print("---------------------------------------")
print(gb_lr_df.to_string(index=False))

print("\nBest learning rate:", best_lr_row["learning_rate"])
print("Best validation RMSE:", round(best_lr_row["val_rmse"], 4))
print("Best validation R2:", round(best_lr_row["val_r2"], 4))






# TASK 7: Cross-validation for the top 2 models

# Combine training and validation data
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

# Define fresh instances of the four candidate models
candidate_models = {
    "linear_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "decision_tree": DecisionTreeRegressor(
        max_depth=6,
        random_state=42
    ),
    "random_forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
}

# Identify the top 2 models using validation RMSE
top_two_models = (
    metrics_df[metrics_df["model"].isin(candidate_models)]
    .nsmallest(2, "RMSE")["model"]
    .tolist()
)

# Use one shared KFold configuration
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = []

for model_name in top_two_models:
    scores = cross_val_score(
        candidate_models[model_name],
        X_train_val,
        y_train_val,
        cv=kf
    )

    cv_results.append({
        "model": model_name,
        "mean_r2": scores.mean(),
        "std_r2": scores.std()
    })

# Save cross-validation results
cv_df = pd.DataFrame(cv_results)
cv_df.to_csv(
    "outputs/part_b_cv.csv",
    index=False
)

print("\nTASK 7: CROSS-VALIDATION")
print("------------------------")
print("Top 2 models selected:", top_two_models)
print("\nCross-validation results:")
print(cv_df.to_string(index=False))



print("\nTASK 7B: FINAL TEST EVALUATION")
print("--------------------------------")

# Select the model with the lowest validation RMSE
final_model_name = (
    metrics_df[metrics_df["split"] == "val"]
    .sort_values("RMSE")
    .iloc[0]["model"]
)

# Use the already-fitted Linear Regression model
final_model = {
    "linear_regression": linear_model,
    "decision_tree": tree_model,
    "random_forest": random_forest_model,
    "gradient_boosting": gb_model
}[final_model_name]

# Predict on the untouched test set
y_test_pred = final_model.predict(X_test)

# Calculate final test metrics
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)

# Add exactly one final test row
test_result = pd.DataFrame([{
    "model": final_model_name,
    "split": "test",
    "MAE": test_mae,
    "MSE": test_mse,
    "RMSE": test_rmse,
    "R2": test_r2
}])

metrics_df = pd.concat(
    [metrics_df, test_result],
    ignore_index=True
)

# Save updated metrics
metrics_df.to_csv(
    "outputs/part_b_metrics.csv",
    index=False
)

print("Final model:", final_model_name)
print("\nFinal test performance:")
print(test_result.to_string(index=False))

print("\nUpdated model metrics:")
print(metrics_df.to_string(index=False))