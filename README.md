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