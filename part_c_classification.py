import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif

from src.shipment_prep import FEATURES

# Task 1 - Stratified Train / Validation / Test Split

# Load modelling dataset
df = pd.read_csv("data/modelling_table.csv")

# Define features and target
X = df[FEATURES]
y = df["is_delayed"]

# First split: 70% training, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3,random_state=42, stratify=y)

# Second split: divide temporary data equally
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Display Task 1 results
print("\nTASK 1: STRATIFIED TRAIN / VALIDATION / TEST SPLIT")
print("-" * 55)

# Display split sizes
print("Original dataset:", X.shape)
print("Training dataset:", X_train.shape)
print("Validation dataset:", X_val.shape)
print("Test dataset:", X_test.shape)

# Display delayed parcel counts
print("\nDelayed parcels in each split:")
print("Training:", int(y_train.sum()))
print("Validation:", int(y_val.sum()))
print("Test:", int(y_test.sum()))

# Display on-time parcel counts
print("\nOn-time parcels in each split:")
print("Training:", int((y_train == 0).sum()))
print("Validation:", int((y_val == 0).sum()))
print("Test:", int((y_test == 0).sum()))





# Task 2 - Blind Baseline and Three Classifiers

# Function to calculate classification metrics
def evaluate_model(model_name, y_actual, y_pred):

    tn, fp, fn, tp = confusion_matrix(y_actual, y_pred, labels=[0, 1]).ravel()

    return {
        "model": model_name,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": accuracy_score(y_actual, y_pred),
        "precision": precision_score(y_actual, y_pred, zero_division=0),
        "recall": recall_score(y_actual, y_pred, zero_division=0),
        "f1": f1_score(y_actual, y_pred, zero_division=0)
    }


# Store model evaluation results
model_results = []


# 1. Blind Baseline - Predict every parcel as on-time
baseline_predictions = [0] * len(y_val)

baseline_result = evaluate_model(
    "blind_all_on_time",
    y_val,
    baseline_predictions
)

model_results.append(baseline_result)


# 2. Logistic Regression
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_val)

logistic_result = evaluate_model(
    "logistic_regression",
    y_val,
    logistic_predictions
)

model_results.append(logistic_result)


# 3. Decision Tree - Find the best max_depth using validation F1
tree_depths = [2, 3, 4, 5, 6, 8, 10]

best_tree_depth = None
best_tree_f1 = -1
best_tree_model = None

for depth in tree_depths:

    tree_model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42
    )

    tree_model.fit(X_train, y_train)

    tree_predictions = tree_model.predict(X_val)

    tree_f1 = f1_score(
        y_val,
        tree_predictions,
        zero_division=0
    )

    # Select highest F1; ties go to smaller depth
    if (
        tree_f1 > best_tree_f1
        or (
            tree_f1 == best_tree_f1
            and (best_tree_depth is None or depth < best_tree_depth)
        )
    ):
        best_tree_f1 = tree_f1
        best_tree_depth = depth
        best_tree_model = tree_model


# Evaluate the selected Decision Tree
best_tree_predictions = best_tree_model.predict(X_val)

tree_result = evaluate_model(
    f"decision_tree_depth_{best_tree_depth}",
    y_val,
    best_tree_predictions
)

model_results.append(tree_result)


# 4. Default Random Forest
random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest_model.fit(X_train, y_train)

random_forest_predictions = random_forest_model.predict(X_val)

random_forest_result = evaluate_model(
    "random_forest_default",
    y_val,
    random_forest_predictions
)

model_results.append(random_forest_result)


# Convert results into a DataFrame
models_df = pd.DataFrame(model_results)

# Save model comparison results
models_df.to_csv(
    "outputs/part_c_models.csv",
    index=False
)


# Display Task 2 results
print("\nTASK 2: BLIND BASELINE AND THREE CLASSIFIERS")
print("-" * 60)

print("\nSelected Decision Tree depth:", best_tree_depth)
print("Selected Decision Tree validation F1:", round(best_tree_f1, 3))

print("\nValidation model comparison:")
print(models_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

print("\nSaved: outputs/part_c_models.csv")






# Task 3 - Feature Importance and Feature Selection

# Import noise columns
from src.shipment_prep import NOISE_COLUMNS

# 1. Random Forest Feature Importance using all 17 features
feature_rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

feature_rf_model.fit(X_train, y_train)

rf_importances = feature_rf_model.feature_importances_


# 2. Calculate ANOVA F-scores using all 17 features
selector_all = SelectKBest(
    score_func=f_classif,
    k="all"
)

selector_all.fit(X_train, y_train)

f_scores = selector_all.scores_


# 3. Create feature scores DataFrame
feature_scores_df = pd.DataFrame({
    "feature": FEATURES,
    "f_score": f_scores,
    "rf_importance": rf_importances,
    "is_noise": [feature in NOISE_COLUMNS for feature in FEATURES]
})

# Rank features by F-score (1 = highest)
feature_scores_df["f_rank"] = (
    feature_scores_df["f_score"]
    .rank(method="min", ascending=False)
    .astype(int)
)

# Arrange columns in the required order
feature_scores_df = feature_scores_df[
    ["feature", "f_score", "f_rank", "rf_importance", "is_noise"]
]

# Sort by F-score rank
feature_scores_df = feature_scores_df.sort_values(
    "f_rank"
).reset_index(drop=True)

# Save feature scores
feature_scores_df.to_csv(
    "outputs/part_c_feature_scores.csv",
    index=False,
    float_format="%.4f"
)


# 4. Test different values of k
k_values = [5, 7, 10, 12, 17]

selectk_results = []

for k in k_values:

    selectk_model = Pipeline([
        ("selector", SelectKBest(f_classif, k=k)),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    selectk_model.fit(X_train, y_train)

    selectk_predictions = selectk_model.predict(X_val)

    selectk_f1 = f1_score(
        y_val,
        selectk_predictions,
        zero_division=0
    )

    selectk_results.append({
        "k": k,
        "val_f1": selectk_f1
    })


# 5. Save feature-selection comparison
selectk_df = pd.DataFrame(selectk_results)

selectk_df.to_csv(
    "outputs/part_c_selectk.csv",
    index=False,
    float_format="%.4f"
)


# Display Task 3 results
print("\nTASK 3: FEATURE IMPORTANCE AND FEATURE SELECTION")
print("-" * 60)

print("\nFeature importance and F-score ranking:")
print(
    feature_scores_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

# Calculate combined importance of noise columns
noise_importance = feature_scores_df.loc[
    feature_scores_df["is_noise"],
    "rf_importance"
].sum()

print("\nCombined Random Forest importance of noise columns:",
      round(noise_importance, 3))

print("\nFeature-selection comparison:")
print(
    selectk_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

print("\nSaved: outputs/part_c_feature_scores.csv")
print("Saved: outputs/part_c_selectk.csv")








# Task 4 - Hyperparameter Tuning

# 1. Select the top 7 features using training data only
selector_k7 = SelectKBest(
    score_func=f_classif,
    k=7
)

X_train_k7 = selector_k7.fit_transform(X_train, y_train)
X_val_k7 = selector_k7.transform(X_val)


# 2. Define Random Forest hyperparameter search space
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10]
}

# Calculate total possible combinations
total_candidates = len(list(ParameterGrid(param_grid)))


# 3. Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5
)

grid_search.fit(X_train_k7, y_train)
grid_predictions = grid_search.predict(X_val_k7)

grid_val_f1 = f1_score(
    y_val,
    grid_predictions,
    zero_division=0
)


# 4. Randomized Search with 5-fold cross-validation
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=9,
    cv=5,
    random_state=42
)

random_search.fit(X_train_k7, y_train)
random_predictions = random_search.predict(X_val_k7)

random_val_f1 = f1_score(
    y_val,
    random_predictions,
    zero_division=0
)


# 5. Default Random Forest using the same 7 features
default_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

default_forest.fit(X_train_k7, y_train)
default_predictions = default_forest.predict(X_val_k7)

default_val_f1 = f1_score(
    y_val,
    default_predictions,
    zero_division=0
)


# 6. Prepare tuning results
tuning_results = [
    {
        "search": "grid_search",
        "candidates": total_candidates,
        "cv_folds": 5,
        "total_fits": total_candidates * 5,
        "best_params": str(grid_search.best_params_),
        "best_cv_accuracy": grid_search.best_score_,
        "val_f1": grid_val_f1
    },
    {
        "search": "randomized_search",
        "candidates": 9,
        "cv_folds": 5,
        "total_fits": 9 * 5,
        "best_params": str(random_search.best_params_),
        "best_cv_accuracy": random_search.best_score_,
        "val_f1": random_val_f1
    },
    {
        "search": "default_forest_same_7_features",
        "candidates": 1,
        "cv_folds": 0,
        "total_fits": 1,
        "best_params": "defaults",
        "best_cv_accuracy": "",
        "val_f1": default_val_f1
    }
]


# 7. Save tuning results
tuning_df = pd.DataFrame(tuning_results)

tuning_df.to_csv(
    "outputs/part_c_tuning.csv",
    index=False,
    float_format="%.4f"
)


# Display Task 4 results
print("\nTASK 4: HYPERPARAMETER TUNING")
print("-" * 60)

print("\nGrid Search best parameters:")
print(grid_search.best_params_)
print("Grid Search best CV accuracy:", round(grid_search.best_score_, 4))

print("\nRandomized Search best parameters:")
print(random_search.best_params_)
print("Randomized Search best CV accuracy:", round(random_search.best_score_, 4))

print("\nValidation F1 comparison:")
print(
    tuning_df[
        ["search", "candidates", "cv_folds", "total_fits", "val_f1"]
    ].to_string(index=False)
)

print("\nComplete tuning results:")
print(tuning_df.to_string(index=False))

print("\nSaved: outputs/part_c_tuning.csv")








# Task 5 - Select Final Model Configuration

# 1. Identify the configuration with the highest validation F1
best_tuning_row = tuning_df.loc[
    tuning_df["val_f1"].idxmax()
]

chosen_from = best_tuning_row["search"]

# 2. Define model parameters based on the selected configuration
if chosen_from == "grid_search":
    final_model_params = grid_search.best_params_

elif chosen_from == "randomized_search":
    final_model_params = random_search.best_params_

else:
    final_model_params = {
        "n_estimators": 100,
        "random_state": 42
    }


# 3. Build the final model pipeline
final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=7)),
    ("model", RandomForestClassifier(
        **final_model_params
    ))
])

# 4. Fit the final model using training data only
final_model.fit(X_train, y_train)


# Display Task 5 results
print("\nTASK 5: FINAL MODEL CONFIGURATION")
print("-" * 60)

print("Selected configuration:", chosen_from)
print("Selected validation F1:", round(best_tuning_row["val_f1"], 4))
print("Final model parameters:", final_model_params)
print("Feature selection: SelectKBest with k=7")
print("Final model fitted using training data only.")










# Task 6 - Operational Alert Threshold

# 1. Get validation probabilities for the delayed class
val_probabilities = final_model.predict_proba(X_val)[:, 1]

# 2. Evaluate different probability thresholds
threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

threshold_results = []

for threshold in threshold_values:

    # Predict delayed if probability >= threshold
    threshold_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_val,
        threshold_predictions,
        labels=[0, 1]
    ).ravel()

    # Calculate evaluation metrics
    threshold_results.append({
        "threshold": threshold,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": accuracy_score(y_val, threshold_predictions),
        "precision": precision_score(
            y_val, threshold_predictions, zero_division=0
        ),
        "recall": recall_score(
            y_val, threshold_predictions, zero_division=0
        ),
        "f1": f1_score(
            y_val, threshold_predictions, zero_division=0
        )
    })


# 3. Create threshold comparison DataFrame
thresholds_df = pd.DataFrame(threshold_results)

# 4. Save threshold results
thresholds_df.to_csv(
    "outputs/part_c_thresholds.csv",
    index=False,
    float_format="%.3f"
)


# 5. Select the highest threshold with recall >= 0.80
eligible_thresholds = thresholds_df[
    thresholds_df["recall"] >= 0.80
]

if not eligible_thresholds.empty:
    chosen_threshold = eligible_thresholds["threshold"].max()
else:
    chosen_threshold = 0.1


# Display Task 6 results
print("\nTASK 6: OPERATIONAL ALERT THRESHOLD")
print("-" * 60)

print("\nThreshold comparison:")
print(
    thresholds_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

print("\nChosen operational threshold:", chosen_threshold)

chosen_row = thresholds_df[
    thresholds_df["threshold"] == chosen_threshold
].iloc[0]

print("Validation precision:", round(chosen_row["precision"], 3))
print("Validation recall:", round(chosen_row["recall"], 3))
print("Validation F1:", round(chosen_row["f1"], 3))

print("\nSaved: outputs/part_c_thresholds.csv")








# Task 7 - Final Test Evaluation

# 1. Generate test probabilities using the final model
test_probabilities = final_model.predict_proba(X_test)[:, 1]

# 2. Apply the selected operational threshold
test_predictions = (
    test_probabilities >= chosen_threshold
).astype(int)

# 3. Calculate test confusion matrix
tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
).ravel()

# 4. Calculate final test metrics
final_test_results = {
    "threshold": chosen_threshold,
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "TP": tp,
    "accuracy": accuracy_score(y_test, test_predictions),
    "precision": precision_score(
        y_test, test_predictions, zero_division=0
    ),
    "recall": recall_score(
        y_test, test_predictions, zero_division=0
    ),
    "f1": f1_score(
        y_test, test_predictions, zero_division=0
    )
}

# 5. Save final test results
final_test_df = pd.DataFrame([final_test_results])

final_test_df.to_csv(
    "outputs/part_c_final_test.csv",
    index=False,
    float_format="%.3f"
)


# 6. Prepare final configuration
final_rf_params = final_model.named_steps["model"].get_params()

max_depth = final_rf_params["max_depth"]

if max_depth is None:
    max_depth = "unlimited"

final_config = {
    "selector_k": 7,
    "n_estimators": final_rf_params["n_estimators"],
    "max_depth": max_depth,
    "min_samples_leaf": final_rf_params["min_samples_leaf"],
    "threshold": chosen_threshold,
    "chosen_from": chosen_from
}

final_config_df = pd.DataFrame([final_config])

final_config_df.to_csv(
    "outputs/part_c_final_config.csv",
    index=False
)


# Display Task 7 results
print("\nTASK 7: FINAL TEST EVALUATION")
print("-" * 60)

print("\nFinal test metrics:")
print(
    final_test_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

print("\nFinal model configuration:")
print(final_config_df.to_string(index=False))

print("\nSaved: outputs/part_c_final_test.csv")
print("Saved: outputs/part_c_final_config.csv")