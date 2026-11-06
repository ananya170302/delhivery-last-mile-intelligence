# Delhivery Last-Mile Delay Intelligence Pipeline

An end-to-end data engineering and machine learning project to analyse last-mile delivery performance, predict shipment delays, segment delivery hubs, and generate operational actions for incoming shipments.

The project uses a synthetic Delhivery-style shipment dataset to demonstrate a reproducible pipeline. It is designed so that the same workflow can be adapted to real operational shipment data.

---

# 1. How to Run

## Prerequisites

- Python 3.14.4
- Git
- pip

## Clone the repository

```bash
git clone https://github.com/ananya170302/delhivery-last-mile-intelligence.git

cd delhivery-last-mile-intelligence
```

## Create and activate a virtual environment

**Git Bash (Windows):**

```bash
python -m venv .venv
source .venv/Scripts/activate
```

**PowerShell (Windows):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the complete pipeline

Execute the scripts in the following order:

```bash
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

## Pipeline overview

| Stage | Description | Main outputs |
|---|---|---|
| Part A – Data preparation and EDA | Clean shipment data, engineer features and analyse delay patterns. | `data/modelling_table.csv`, `outputs/part_a_checks.csv`, `outputs/part_a_delay_by_group.csv`, `outputs/part_a_correlations.csv` |
| Part B – Regression | Predict delivery duration and compare regression models. | Regression metrics and model comparison outputs under `outputs/` |
| Part C – Classification | Predict shipment delays, select features, tune models and determine the alert threshold. | Classification metrics, feature selection, tuning results and `outputs/part_c_final_config.csv` |
| Part D – Hub segmentation | Profile hubs, perform K-Means clustering and compare hub tiers against performance-based segments. | `outputs/hub_profile.csv`, `outputs/part_d_separation.csv`, `outputs/hub_segments.csv`, `outputs/segment_profile.csv`, `outputs/tier_vs_segment_mismatch.csv` |
| Model packaging | Rebuild and serialize the final classification pipeline. | `models/delay_model.pkl`, `models/delay_threshold.csv`, `models/feature_order.csv`, `models/impute_values.csv` |
| New shipment scoring | Score incoming shipments and generate operational actions. | `outputs/new_shipment_actions.csv` |

**Model packaging note:** `package_model.py` rebuilds the final classification pipeline using the saved Part C configuration and regenerates the model artifacts. The serialized model depends on the installed scikit-learn version. Use the pinned dependencies in `requirements.txt` when loading the committed pickle. If the installed version differs, rerun `package_model.py` before running `predict_delay.py`.

---

# 2. Data Findings

Based on the analysis of 1,200 historical shipments, the following patterns were identified.

## 2.1 Service type and delivery delays

Express shipments have the highest delay rate at **48.97%** (119 delayed shipments out of 243).

Standard shipments have a delay rate of **20.56%**, while Economy shipments have the lowest rate of **8.57%**.

This indicates that Express deliveries require closer monitoring despite having the shortest promised delivery window.

## 2.2 Hub tier and delivery delays

The delay rate increases significantly across hub tiers:

| Hub Tier | Delay Rate |
|---|---:|
| Tier 1 | 4.00% |
| Tier 2 | 21.50% |
| Tier 3 | 44.00% |

This indicates a strong relationship between hub tier and delivery delays, highlighting Tier 3 hubs as an area requiring further operational investigation.

## 2.3 Distance and delivery delays

A clear increase in delay rates is observed as shipment distance increases.

| Distance Band | Delay Rate |
|---|---:|
| 0–200 km | 2.19% |
| 201–400 km | 13.68% |
| 401–600 km | 43.64% |
| Above 600 km | 85.23% |

Longer delivery routes therefore represent a significant operational risk.

## 2.4 Factors correlated with delivery time

The three strongest correlations with `delivery_hours`, excluding the target itself, are:

| Feature | Correlation |
|---|---:|
| `promised_hours` | 0.8372 |
| `distance_km` | 0.4013 |
| `n_stops` | 0.3247 |

These positive correlations indicate that longer promised delivery windows, greater travel distances, and more delivery stops are associated with longer actual delivery times.

Correlation does not establish causation.

## 2.5 Declared-value transformation

The median-filled `declared_value_inr` column has:

| Statistic | Original Value | Log-transformed Value |
|---|---:|---:|
| Mean | 1533.0542 | 7.0277 |
| Median | 1121.50 | 7.0224 |

The smaller difference between the transformed mean and median suggests that the log transformation reduces the influence of high-value observations and makes the feature's distribution less right-skewed.

## 2.6 Data preparation summary

The historical dataset initially contained 1,224 records, including 24 duplicate records. After duplicate removal, 1,200 unique shipment records remained.

Data preparation included:

- Removing duplicate shipment records.
- Standardizing inconsistent service type values.
- Handling missing payment modes using `Unknown`.
- Converting declared values into numeric INR values.
- Parsing dispatch timestamps.
- Treating parcel weights above 100 kg as invalid.
- Imputing missing numerical values using historical medians.

The historical medians used for imputation were:

| Column | Median |
|---|---:|
| `distance_km` | 314.95 |
| `parcel_weight_kg` | 1.83 |
| `declared_value_inr` | 1121.50 |

The cleaned dataset was enriched with hub information and additional features such as promised delivery hours, COD indicator, weekend indicator, dispatch hour and log-transformed declared value.

The resulting modelling table contains 1,200 shipments, with 278 delayed shipments and an overall delay rate of approximately **23.17%**.

## 2.7 Evidence and source files

The findings above are supported by the following project outputs:

| Finding | Evidence file | Related task |
|---|---|---|
| Service type delay rates | `outputs/part_a_delay_by_group.csv` | Task 6 – Delay analysis by group |
| Hub tier delay rates | `outputs/part_a_delay_by_group.csv` | Task 6 – Delay analysis by group |
| Distance-band delay rates | `outputs/part_a_delay_by_group.csv` | Task 6 – Delay analysis by group |
| Top three correlations with delivery time | `outputs/part_a_correlations.csv` | Task 7 – Correlation analysis |
| Declared-value mean, median and log-transformation statistics | `outputs/part_a_checks.csv` | Task 5 – Part A validation checks |

**Additional supporting files:**

- `data/modelling_table.csv` – Final cleaned dataset with engineered features and the delay target, used for analysis and modelling.
- `part_a_clean_eda.py` – Script that generates the modelling table and Part A analysis outputs.
- `src/shipment_prep.py` – Contains the shared data-cleaning and feature-engineering functions.

---

# 3. Model Results

Two machine learning problems were developed:

1. Regression – Predicting delivery duration in hours.
2. Classification – Predicting whether a shipment will be delayed.

## 3.1 Delivery duration prediction – Regression

The objective of Part B was to predict the actual delivery duration of a shipment using its operational and shipment-level features.

The target variable was `delivery_hours`.

### 3.1.1 Train / Validation / Test split

The dataset was divided into training, validation and testing sets using a fixed random state of 42.

| Dataset | Records |
|---|---:|
| Training | 840 |
| Validation | 180 |
| Testing | 180 |
| **Total** | **1,200** |

The scaler was fitted only on the training data to prevent data leakage.

### 3.1.2 Baseline performance

The mean baseline model achieved a validation R² of **-0.0212** with an RMSE of **18.8552 hours**.

The negative R² indicates that simply predicting the average delivery time performs worse than using the validation dataset's actual mean as a reference.

This confirms that a baseline prediction is insufficient and that the shipment features provide useful predictive information.

### 3.1.3 Model comparison

Four candidate models were evaluated using the validation dataset.

| Model | Validation MAE | Validation RMSE | Validation R² |
|---|---:|---:|---:|
| Linear Regression | 3.9222 | 4.8912 | 0.9313 |
| Decision Tree | 5.0971 | 6.2796 | 0.8867 |
| Random Forest | 4.2505 | 5.4680 | 0.9141 |
| Gradient Boosting | 3.9166 | 4.9165 | 0.9306 |

**Linear Regression was selected as the final model** because it achieved the lowest validation RMSE of **4.8912 hours**.

Gradient Boosting performed very closely, with a validation RMSE of 4.9165 hours.

### 3.1.4 Decision Tree depth analysis

The tree-depth experiment showed how model complexity affects training and validation performance.

- **High bias – depth 1:** Training RMSE was 11.6334 and validation RMSE was 12.1324. Both errors were relatively high, indicating that the model was too simple to capture the underlying patterns.
- **Best balance – depth 6:** Training RMSE was 4.7114 and validation RMSE was 6.2796. This depth provided the lowest validation RMSE among the tested tree configurations.
- **High variance – unlimited depth:** Training RMSE reached 0.0000, while validation RMSE increased to 7.7890. The large gap indicates overfitting: the tree learned the training data extremely well but generalized less effectively to unseen data.

Depth 6 was therefore selected for the Decision Tree candidate.

### 3.1.5 Gradient Boosting learning-rate experiment

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

### 3.1.6 Cross-validation and final model selection

The two best validation models were evaluated using 5-fold cross-validation on the combined training and validation datasets.

| Model | Mean R² | Std R² |
|---|---:|---:|
| Linear Regression | 0.9261 | 0.0048 |
| Gradient Boosting | 0.9352 | 0.0044 |

Gradient Boosting achieved a slightly higher mean cross-validation R². However, Linear Regression had the lowest validation RMSE.

Since the final model selection criterion was the lowest validation RMSE, **Linear Regression was selected**.

### 3.1.7 Final test performance

The selected Linear Regression model was evaluated on the untouched test dataset.

| Metric | Test Result |
|---|---:|
| MAE | 3.5671 hours |
| MSE | 20.4624 |
| RMSE | 4.5235 hours |
| R² | 0.9493 |

The model achieved a test R² of **0.9493**, indicating that it explains approximately 94.93% of the variation in delivery hours on the test dataset.

The test RMSE of 4.5235 hours is lower than the validation RMSE of 4.8912 hours. Along with the test R² of 0.9493, this indicates strong performance on the held-out test dataset.

---

## 3.2 Shipment delay prediction – Classification

The objective of Part C was to build a classification model that predicts whether a shipment will be delayed.

The target variable is `is_delayed`:

- `1` = Delayed shipment
- `0` = On-time shipment

The classification model uses the same 17 pinned features established in Part A, followed by feature selection to identify the most useful predictors.

### 3.2.1 Stratified train / validation / test split

The dataset was divided using stratified sampling to preserve the proportion of delayed and on-time shipments.

| Dataset | Total Rows | Delayed | On-time |
|---|---:|---:|---:|
| Training | 840 | 195 | 645 |
| Validation | 180 | 42 | 138 |
| Test | 180 | 41 | 139 |
| **Total** | **1,200** | **278** | **922** |

The test dataset was kept separate from model training, feature selection and threshold tuning.

### 3.2.2 Blind baseline and classifier comparison

A blind baseline was created by predicting every shipment as on-time, without considering any input features.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Blind baseline | 0.767 | 0.000 | 0.000 | 0.000 |
| Logistic Regression | 0.906 | 0.778 | 0.833 | 0.805 |
| Decision Tree (depth 6) | 0.883 | 0.756 | 0.738 | 0.747 |
| Random Forest (default) | 0.906 | 0.838 | 0.738 | 0.785 |

The blind baseline achieved 76.7% accuracy simply because most shipments in the dataset were on-time. However, it failed to identify a single delayed shipment.

This demonstrates why accuracy alone is not sufficient for an imbalanced classification problem. Precision, recall and F1 are also important when evaluating a delay prediction model.

All three trained classifiers performed better than the blind baseline.

### 3.2.3 Feature importance and selection

Two approaches were used to evaluate feature importance:

- ANOVA F-scores using `SelectKBest(f_classif)`.
- Random Forest feature importance using a default Random Forest model.

The top seven features based on ANOVA F-score were:

| Rank | Feature | F-score |
|---:|---|---:|
| 1 | `distance_km` | 389.2131 |
| 2 | `hub_tier` | 155.3483 |
| 3 | `hub_daily_capacity` | 144.1347 |
| 4 | `n_stops` | 125.4798 |
| 5 | `promised_hours` | 84.4177 |
| 6 | `rain_flag` | 53.0321 |
| 7 | `sale_event_flag` | 36.7281 |

The five noise columns were:

- `label_print_batch`
- `scanner_build_no`
- `tape_roll_no`
- `sticker_serial_mod`
- `shelf_slot_no`

All five noise columns ranked below the top seven and had F-scores below 10.

Their combined Random Forest feature importance was approximately 0.224 (22.4%). This indicates that the noise features collectively received some model importance despite having weak individual statistical relationships with the target.

#### Feature selection comparison

Different values of k were evaluated using a Random Forest model with 100 estimators.

| Selected Features (k) | Validation F1 |
|---:|---:|
| 5 | 0.6988 |
| 7 | 0.8333 |
| 10 | 0.7857 |
| 12 | 0.8049 |
| 17 | 0.7848 |

Using seven features achieved the highest validation F1 of 0.8333. Therefore, `k=7` was fixed for the remaining classification experiments.

### 3.2.4 Hyperparameter tuning

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

### 3.2.5 Operational alert threshold

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

### 3.2.6 Final test evaluation

The final model was evaluated once on the untouched test dataset using the selected threshold of 0.5.

| Metric | Test Result |
|---|---:|
| Accuracy | 0.867 |
| Precision | 0.689 |
| Recall | 0.756 |
| F1 | 0.721 |

#### Test confusion matrix

| | Predicted On-time | Predicted Delayed |
|---|---:|---:|
| Actually On-time | 125 | 14 |
| Actually Delayed | 10 | 31 |

Out of 41 delayed shipments in the test dataset, the model correctly identified 31 and missed 10.

The validation recall was 0.833, while test recall was 0.756. This indicates a drop in delayed-shipment detection performance on unseen data.

The test dataset contains only 41 delayed shipments, so each additional missed shipment has a noticeable effect on recall.

### 3.2.7 Final model configuration

| Parameter | Selected Value |
|---|---|
| Feature Selection | SelectKBest |
| Number of Features | 7 |
| Model | Random Forest Classifier |
| Number of Estimators | 100 |
| Maximum Depth | Unlimited (default) |
| Minimum Samples per Leaf | 1 |
| Alert Threshold | 0.5 |
| Selected Configuration | `default_forest_same_7_features` |

### 3.2.8 Key findings

1. The blind baseline achieved 76.7% accuracy but failed to identify delayed shipments, highlighting the importance of evaluating recall and F1 alongside accuracy.
2. Distance, hub characteristics, number of stops, promised delivery time, rain and sale events were the strongest features based on ANOVA F-scores.
3. Selecting seven features improved validation F1 to 0.8333 compared with using all 17 features, which achieved 0.7848.
4. Grid Search and Randomized Search did not outperform the default Random Forest on validation F1.
5. A threshold of 0.5 provided a validation recall of 0.833 while maintaining precision of 0.795.
6. The final test F1 was 0.721, with recall of 0.756, showing that model performance decreased on unseen data.

---

# 4. Operations Recommendation

## 4.1 Hub profiling and segmentation

The objective of Part D was to group delivery hubs based on their actual delivery performance rather than relying only on predefined hub tiers.

The 24 hubs were profiled using the following metrics:

- Total shipments
- Mean delivery hours
- Delay rate
- Mean distance
- Mean number of stops

Each hub had 50 historical shipments.

The four hub-level measurements used for clustering were standardized using `StandardScaler`:

- `mean_delivery_hours`
- `delay_rate`
- `mean_distance_km`
- `mean_n_stops`

K-Means was configured with:

- `n_clusters=3`
- `random_state=42`
- `n_init=10`

Clusters were ordered by their average hub delay rate and assigned operational names.

## 4.2 Final hub segments

| Segment | Number of Hubs | Average Delivery Hours | Delay Rate | Average Distance (km) | Average Stops |
|---|---:|---:|---:|---:|---:|
| Standard | 7 | 47.24 | 2.57% | 188.93 | 55.37 |
| Watch | 8 | 56.81 | 15.50% | 314.18 | 70.72 |
| Escalate | 9 | 64.65 | 46.00% | 470.48 | 84.34 |

### Standard

These hubs have the lowest average delivery time and delay rate.

Recommended action:

- Maintain current operations.
- Continue routine monitoring.
- Use these hubs as a performance benchmark for other hubs.

### Watch

These hubs have moderate delivery times and delay rates.

Recommended action:

- Review daily delay trends.
- Monitor route performance and shipment volumes.
- Investigate changes in capacity utilization.
- Consider proactive customer communication for shipments flagged by the classification model.

### Escalate

These hubs have the highest average delivery times and delay rates.

Recommended action:

- Prioritize operational intervention.
- Escalate high-risk shipments to hub managers.
- Review capacity, route distance and delivery stop counts.
- Investigate recurring delays and resource constraints.

The Escalate segment includes:

- HUB13 – Nagpur
- HUB15 – Kochi
- HUB17 – Patna
- HUB18 – Ranchi
- HUB19 – Guwahati
- HUB20 – Raipur
- HUB21 – Bhubaneswar
- HUB23 – Siliguri
- HUB24 – Agartala

## 4.3 Choosing three clusters

The across-to-within distance ratio was calculated for K=2, 3, 4 and 5 to evaluate cluster separation.

| K | Across-to-within Ratio | Cluster Sizes |
|---|---:|---|
| 2 | 2.894 | 15-9 |
| 3 | 5.393 | 9-8-7 |
| 4 | 6.305 | 8-7-5-4 |
| 5 | 6.887 | 8-7-4-3-2 |

Although K=4 and K=5 produced higher separation ratios, they also created smaller clusters containing as few as two hubs.

K=3 was selected because it provides a useful balance between cluster separation and operational manageability. Each segment has enough hubs to support a meaningful operational playbook.

## 4.4 Hub tier versus actual performance

The expected segment based on hub tier was compared against the actual performance-based segment.

Expected mapping:

| Hub Tier | Expected Segment |
|---|---|
| Tier 1 | Standard |
| Tier 2 | Watch |
| Tier 3 | Escalate |

The following mismatches were identified:

| Hub ID | City | Hub Tier | Actual Segment |
|---|---|---|---|
| HUB07 | Kolkata | Tier 1 | Watch |
| HUB13 | Nagpur | Tier 2 | Escalate |
| HUB15 | Kochi | Tier 2 | Escalate |
| HUB22 | Dehradun | Tier 3 | Watch |

These mismatches are worth investigating.

A hub's predefined tier does not necessarily reflect its actual delivery performance. Capacity, distance, route conditions and operational processes should also be reviewed.

## 4.5 New shipment delay predictions and actions

The packaged classification model was used to score 48 incoming shipments from `data/new_shipments.csv`.

The prediction pipeline:

1. Loads incoming shipment records.
2. Applies the shared cleaning logic.
3. Uses the historical imputation medians saved during model packaging.
4. Generates the same engineered features used during model training.
5. Arranges features in the saved model feature order.
6. Calculates delay probabilities.
7. Flags shipments when the probability is greater than or equal to 0.50.
8. Joins the shipment's hub segment.
9. Assigns an operational action.

### Prediction results

| Action | Shipment Count |
|---|---:|
| No action | 40 |
| Proactive customer call | 2 |
| Escalate to hub manager | 6 |
| **Total** | **48** |

Out of 48 incoming shipments:

- 8 shipments were flagged as potentially delayed.
- 40 shipments were not flagged.
- 6 flagged shipments belonged to the Escalate segment.
- 2 flagged shipments belonged to other segments.

### Operational decision logic

| Condition | Action |
|---|---|
| Shipment is not flagged | No action |
| Shipment is flagged and hub segment is Escalate | Escalate to hub manager |
| Shipment is flagged and hub segment is Standard or Watch | Proactive customer call |

This connects shipment-level risk predictions with hub-level operational performance.

The resulting output is saved in:

`outputs/new_shipment_actions.csv`

The output contains:

- `shipment_id`
- `hub_id`
- `segment`
- `delay_probability`
- `flagged`
- `action`

The prediction script was also executed twice, and the resulting CSV files were byte-identical, confirming deterministic output for the same input and model artifacts.

---

# 5. Production Plan

The current project demonstrates the foundation of a production-oriented data and machine learning pipeline. The following characteristics and improvements would help operationalize it.

## 5.1 Production characteristics

| Characteristic | Implementation or proposed approach |
|---|---|
| Accuracy | Evaluate regression and classification models using MAE, RMSE, R², precision, recall and F1-score. Reassess performance using newly labelled shipment data. |
| Reliability | Validate incoming data, check missing values, verify hub mappings and stop processing when required hub segments are unavailable. |
| Reproducibility | Use fixed random states, pinned dependencies, deterministic data generation, shared preparation functions and saved model configuration. |
| Performance | Use vectorized pandas operations and batch prediction rather than processing shipments individually. |
| Maintainability | Keep data preparation in `src/shipment_prep.py` and separate data analysis, training, packaging and prediction into individual scripts. |
| Consistency | Reuse the same cleaning logic, historical imputation medians, feature order and saved alert threshold during inference. |
| Monitorability | Track shipment volumes, missing values, delay rates, model predictions, flagged shipments and hub-level performance. |

## 5.2 Monitoring and alert thresholds

The following are proposed operational monitoring rules for a future production deployment. They are not thresholds validated by the current synthetic dataset.

| Monitoring Metric | Proposed Trigger | Recommended Response |
|---|---|---|
| Missing values in key input fields | Above 5% | Investigate source data quality and alert the pipeline owner. |
| Hub segment mapping | Any shipment without a segment | Stop scoring and investigate the hub reference data. |
| Flagged shipment rate | Above 25% in a scoring batch | Review shipment mix, hub performance and model output distribution. |
| Hub delay rate | Increase of more than 10 percentage points against its rolling four-week baseline | Trigger an operational hub review. |
| Pipeline output count | Scored shipment count differs from valid incoming shipment count | Investigate dropped or duplicated records. |

## 5.3 Retraining strategy

A practical model maintenance schedule would include:

- **Monthly monitoring:** Evaluate model performance using newly completed and labelled shipments.
- **Quarterly retraining:** Retrain the model using updated historical shipment data.
- **Drift-based retraining:** Consider earlier retraining if significant changes in input distributions or model performance are observed.
- **Threshold review:** Reassess the alert threshold when operational priorities or the cost of false positives and false negatives change.

The retraining process should also compare the new model against the existing model before replacing it.

## 5.4 Seasonal and sale-event handling

Sale periods can change shipment volumes and delivery patterns significantly.

For the sale-event window of **15–28 September**, performance should be compared with similar historical sale-event periods rather than ordinary operating weeks.

The production pipeline should retain event indicators and historical records so that seasonal performance can be evaluated separately.

For example, a September sale period should ideally be compared against previous September sale periods, rather than a normal operating period with lower shipment volumes.

## 5.5 Future production architecture

A production implementation could extend the current workflow with:

1. **Data ingestion:** Retrieve shipment and hub data from operational systems or APIs.
2. **Data validation:** Validate schemas, required fields, duplicate records and missing values.
3. **Data preparation:** Apply the shared cleaning and feature-engineering logic.
4. **Model inference:** Load the packaged model and score shipments in batches.
5. **Operational decisioning:** Combine delay probabilities with hub segments to determine actions.
6. **Alert delivery:** Send escalations to hub managers and customer-call tasks to the relevant teams.
7. **Monitoring:** Track pipeline health, data quality, prediction distributions and actual delivery outcomes.
8. **Model lifecycle:** Retrain, validate and version models as new labelled data becomes available.

---

# 6. Limitations

1. **Synthetic dataset:** The project uses generated shipment data rather than a live Delhivery operational feed. Real-world delivery patterns may differ.

2. **Limited dataset size:** The historical dataset contains 1,200 shipments, and the final classification test set contains only 41 delayed shipments. Performance estimates may change with larger datasets.

3. **Threshold generalization:** The alert threshold was selected using a single validation split. It should be reassessed using additional historical and operational data.

4. **Limited operational context:** The current dataset does not provide live traffic, real-time weather, road closures or detailed route-level information.

5. **Clustering sensitivity:** Hub segments depend on the selected hub-level features, scaling method and number of clusters. Segments may change as more operational data becomes available.

6. **Prediction uncertainty:** A predicted delay probability is a risk estimate, not a guarantee that a shipment will be delayed. False positives and false negatives remain possible.

7. **Production integration:** The current project produces CSV-based outputs. A live deployment would require integration with shipment systems, scheduled execution, monitoring, alert delivery and access controls.

8. **Seasonal generalization:** The synthetic dataset does not establish how well the model performs across multiple years or different sale-event periods. More historical seasonal data would be needed to validate event-specific performance.

---

**Project repository:** https://github.com/ananya170302/delhivery-last-mile-intelligence

**Note:** The dataset used in this project is synthetic and does not represent a live Delhivery operational feed. The results demonstrate the proposed data engineering, machine learning and operational decision-making workflow and should not be interpreted as production-level delivery performance.