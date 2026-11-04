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