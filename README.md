## Data findings

Based on the analysis of 1,200 historical shipments, the following patterns were identified:

1. **Service type and delivery delays:** Express shipments have the highest delay rate at **48.97%** (119 delayed shipments out of 243). Standard shipments have a delay rate of **20.56%**, while Economy shipments have the lowest rate at **8.57%**. This indicates that Express deliveries require closer monitoring despite having the shortest promised delivery window.

2. **Hub tier and delivery delays:** The delay rate increases significantly across hub tiers, from **4.00% for Tier 1**, to **21.50% for Tier 2**, and **44.00% for Tier 3**. This indicates a strong relationship between hub tier and delivery delays, highlighting Tier 3 hubs as an area requiring further operational investigation.

3. **Distance and delivery delays:** A clear increase in delay rates is observed as shipment distance increases. The rates are **2.19%** for 0–200 km, **13.68%** for 201–400 km, **43.64%** for 401–600 km, and **85.23%** for shipments above 600 km. Longer delivery routes therefore represent a significant operational risk.

4. **Factors correlated with delivery time:** The three strongest correlations with `delivery_hours`, excluding the target itself, are `promised_hours` (**0.8372**), `distance_km` (**0.4013**), and `n_stops` (**0.3247**). These positive correlations indicate that longer promised delivery windows, greater travel distances, and more delivery stops are associated with longer actual delivery times. Correlation does not establish causation.

5. **Declared-value transformation:** The median-filled `declared_value_inr` column has a mean of **INR 1,533.0542** and a median of **INR 1,121.50**, indicating a right-skewed distribution. After applying the natural logarithm, the mean is **7.0277** and the median is **7.0224**. The smaller difference between the transformed mean and median suggests that the log transformation reduces the influence of high-value observations and makes the feature's distribution less right-skewed.

## Evidence and source files

The findings above are supported by the following project outputs:  

| Finding | Evidence file | Related task |
|---------|---------------|--------------|
| Service type delay rates | `outputs/part_a_delay_by_group.csv` | Task 6 – Delay analysis by group |
| Hub tier delay rates | `outputs/part_a_delay_by_group.csv` | Task 6 – Delay analysis by group |
| Distance-band delay rates | `outputs/part_a_delay_by_group.csv` | Task 6 – Delay analysis by group |
| Top three correlations with delivery time | `outputs/part_a_correlations.csv` | Task 7 – Correlation analysis |
| Declared-value mean, median and log-transformation statistics | `outputs/part_a_checks.csv` | Task 5 – Part A validation checks |

**Additional supporting files:**

- `data/modelling_table.csv` – Final cleaned dataset with engineered features and the delay target, used for the analysis.
- `part_a_clean_eda.py` – Script that generates the modelling table and Part A analysis outputs.
- `src/shipment_prep.py` – Contains the data-cleaning and feature-engineering functions.







## Model results (regression)

### 1. Baseline performance

The mean baseline model achieved a validation R² of **-0.0212** with an RMSE of **18.8552 hours**.

The negative R² indicates that simply predicting the average delivery time performs worse than using the validation dataset's actual mean as a reference. This confirms that a baseline prediction is insufficient and that the shipment features provide useful predictive information.

### 2. Model comparison

Four candidate models were evaluated using the validation dataset.

| Model | Validation MAE | Validation RMSE | Validation R² |
|---|---:|---:|---:|
| Linear Regression | 3.9222 | 4.8912 | 0.9313 |
| Decision Tree | 5.0971 | 6.2796 | 0.8867 |
| Random Forest | 4.2505 | 5.4680 | 0.9141 |
| Gradient Boosting | 3.9166 | 4.9165 | 0.9306 |

**Linear Regression was selected as the final model** because it achieved the lowest validation RMSE of **4.8912 hours**.

Gradient Boosting performed very closely, with a validation RMSE of 4.9165 hours.

### 3. Decision Tree depth analysis

The tree-depth experiment showed how model complexity affects training and validation performance.

- **High bias — depth 1:** Training RMSE was 11.6334 and validation RMSE was 12.1324. Both errors were relatively high, indicating that the model was too simple to capture the underlying patterns.
- **Best balance — depth 6:** Training RMSE was 4.7114 and validation RMSE was 6.2796. This depth provided the lowest validation RMSE among the tested tree configurations.
- **High variance — unlimited depth:** Training RMSE reached 0.0000, while validation RMSE increased to 7.7890. The large gap indicates overfitting: the tree learned the training data extremely well but generalized less effectively to unseen data.

Depth 6 was therefore selected for the Decision Tree candidate.

### 4. Gradient Boosting learning-rate experiment

Three learning rates were evaluated while keeping the number of estimators fixed at 100.

| Learning Rate | Validation RMSE | Validation R² |
|---|---:|---:|
| 0.01 | 10.0018 | 0.7127 |
| 0.1 | 4.9165 | 0.9306 |
| 1.0 | 7.3701 | 0.8440 |

A learning rate of **0.1** produced the best performance.

- **0.01:** Learning was too slow, and 100 estimators were insufficient to capture the patterns effectively.
- **0.1:** Provided a good balance between learning speed and model performance.
- **1.0:** The larger updates resulted in less stable learning and poorer validation performance.

### 5. Cross-validation and final model selection

The two best validation models were evaluated using 5-fold cross-validation on the combined training and validation datasets.

| Model | Mean R² | Std R² |
|---|---:|---:|
| Linear Regression | 0.9261 | 0.0048 |
| Gradient Boosting | 0.9352 | 0.0044 |

Gradient Boosting achieved a slightly higher mean cross-validation R². However, Linear Regression had the lowest validation RMSE.

Since the final model selection criterion was the lowest validation RMSE, **Linear Regression was selected**.

### 6. Final test performance

The selected Linear Regression model was evaluated on the untouched test dataset.

| Metric | Test Result |
|---|---:|
| MAE | 3.5671 hours |
| MSE | 20.4624 |
| RMSE | 4.5235 hours |
| R² | 0.9493 |

The model achieved a test R² of **0.9493**, indicating that it explains approximately 94.93% of the variation in delivery hours on the test dataset.

The test RMSE of 4.5235 hours is lower than the validation RMSE of 4.8912 hours. Along with the test R² of 0.9493, this indicates that the selected model generalizes well to the held-out test dataset.







## Model Results (Classification)

### 1. Classification Objective

The objective of Part C was to build a classification model that predicts whether a shipment will be delayed.

The target variable is `is_delayed`:
- `1` = Delayed shipment
- `0` = On-time shipment

The classification model uses the same 17 pinned features established in Part A, followed by feature selection to identify the most useful predictors.

### 2. Stratified Train / Validation / Test Split

The dataset was divided using stratified sampling to preserve the proportion of delayed and on-time shipments.

| Dataset | Total Rows | Delayed | On-time |
|---|---:|---:|---:|
| Training | 840 | 195 | 645 |
| Validation | 180 | 42 | 138 |
| Test | 180 | 41 | 139 |
| **Total** | **1,200** | **278** | **922** |

The test dataset was kept separate from model training, feature selection, and threshold tuning.

### 3. Blind Baseline and Classifier Comparison

A blind baseline was created by predicting every shipment as on-time, without considering any input features.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Blind baseline | 0.767 | 0.000 | 0.000 | 0.000 |
| Logistic Regression | 0.906 | 0.778 | 0.833 | 0.805 |
| Decision Tree (depth 6) | 0.883 | 0.756 | 0.738 | 0.747 |
| Random Forest (default) | 0.906 | 0.838 | 0.738 | 0.785 |

The blind baseline achieved 76.7% accuracy simply because most shipments in the dataset were on-time. However, it failed to identify a single delayed shipment.

This demonstrates why accuracy alone is not sufficient for an imbalanced classification problem. Precision, recall, and F1 are also important when evaluating a delay prediction model.

All three trained classifiers performed better than the blind baseline.

### 4. Feature Importance and Selection

Two approaches were used to evaluate feature importance:

- ANOVA F-scores using `SelectKBest(f_classif)`.
- Random Forest feature importance using a default Random Forest model.

The top seven features based on ANOVA F-score were:

| Rank | Feature | F-score |
|---:|---|---:|
| 1 | distance_km | 389.2131 |
| 2 | hub_tier | 155.3483 |
| 3 | hub_daily_capacity | 144.1347 |
| 4 | n_stops | 125.4798 |
| 5 | promised_hours | 84.4177 |
| 6 | rain_flag | 53.0321 |
| 7 | sale_event_flag | 36.7281 |

The five noise columns were:
- label_print_batch
- scanner_build_no
- tape_roll_no
- sticker_serial_mod
- shelf_slot_no

All five noise columns ranked below the top seven and had F-scores below 10.

Their combined Random Forest feature importance was approximately 0.224 (22.4%). This indicates that the noise features collectively received some model importance despite having weak individual statistical relationships with the target.

#### Feature Selection Comparison

Different values of k were evaluated using a Random Forest model with 100 estimators.

| Selected Features (k) | Validation F1 |
|---:|---:|
| 5 | 0.6988 |
| 7 | 0.8333 |
| 10 | 0.7857 |
| 12 | 0.8049 |
| 17 | 0.7848 |

Using seven features achieved the highest validation F1 of 0.8333. Therefore, `k=7` was fixed for the remaining classification experiments.

### 5. Hyperparameter Tuning

Grid Search and Randomized Search were performed using the seven selected features.

The search space included:

- `n_estimators`: 50, 100, 150
- `max_depth`: 4, 6, 8
- `min_samples_leaf`: 1, 5, 10

Both approaches used five-fold cross-validation.

| Search Method | Candidates | Total Fits | Best CV Accuracy | Validation F1 |
|---|---:|---:|---:|---:|
| Grid Search | 27 | 135 | 0.8952 | 0.7750 |
| Randomized Search | 9 | 45 | 0.8905 | 0.7848 |
| Default Random Forest | 1 | 1 | N/A | 0.8333 |

Grid Search achieved slightly higher cross-validation accuracy than Randomized Search.

However, the default Random Forest achieved the highest validation F1 of 0.8333.

This shows that additional hyperparameter tuning did not improve validation F1 in this experiment. The tuned searches optimized cross-validation accuracy, whereas the final configuration was selected using validation F1.

The default Random Forest configuration was therefore selected.

### 6. Operational Alert Threshold

The final model's predicted probabilities were evaluated using thresholds from 0.1 to 0.9.

The operational requirement was to select the highest threshold that maintained validation recall of at least 0.80.

The selected threshold was **0.5**.

At this threshold:

| Metric | Validation Result |
|---|---:|
| Accuracy | 0.911 |
| Precision | 0.795 |
| Recall | 0.833 |
| F1 | 0.814 |

At a threshold of 0.6, recall dropped to 0.762, which was below the required 0.80. Therefore, 0.5 was the highest qualifying threshold.

### 7. Final Test Evaluation

The final model was evaluated once on the untouched test dataset using the selected threshold of 0.5.

| Metric | Test Result |
|---|---:|
| Accuracy | 0.867 |
| Precision | 0.689 |
| Recall | 0.756 |
| F1 | 0.721 |

#### Test Confusion Matrix

| | Predicted On-time | Predicted Delayed |
|---|---:|---:|
| Actually On-time | 125 | 14 |
| Actually Delayed | 10 | 31 |

Out of 41 delayed shipments in the test dataset, the model correctly identified 31 and missed 10.

The validation recall was 0.833, while test recall was 0.756. This indicates a drop in delayed-shipment detection performance on unseen data.

The test dataset contains only 41 delayed shipments, so each additional missed shipment has a noticeable effect on recall.

### 8. Final Model Configuration

| Parameter | Selected Value |
|---|---|
| Feature Selection | SelectKBest |
| Number of Features | 7 |
| Model | Random Forest Classifier |
| Number of Estimators | 100 |
| Maximum Depth | Unlimited (default) |
| Minimum Samples per Leaf | 1 |
| Alert Threshold | 0.5 |
| Selected Configuration | default_forest_same_7_features |

### 9. Key Findings

1. The blind baseline achieved 76.7% accuracy but failed to identify delayed shipments, highlighting the importance of evaluating recall and F1 alongside accuracy.
2. Distance, hub characteristics, number of stops, promised delivery time, rain, and sale events were the strongest features based on ANOVA F-scores.
3. Selecting seven features improved validation F1 to 0.8333 compared with using all 17 features, which achieved 0.7848.
4. Grid Search and Randomized Search did not outperform the default Random Forest on validation F1.
5. A threshold of 0.5 provided a validation recall of 0.833 while maintaining precision of 0.795.
6. The final test F1 was 0.721, with recall of 0.756, showing that model performance decreased on unseen data.

**Note:** The dataset used in this project is synthetic and does not represent a live Delhivery operational feed. The results demonstrate the proposed classification workflow and should not be interpreted as production-level delivery performance.