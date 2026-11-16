# EV Battery Degradation: Predicting State of Health (SoH)

## Overview
This project predicts the State of Health (`SoH_Percent`) of electric vehicle
batteries from usage and vehicle data, using regression models. A
Linear Regression and a Random Forest Regressor are trained and compared.

## Dataset
- Source: [Kaggle – EV Battery Degradation and Charge](https://www.kaggle.com/datasets/bertnardomariouskono/electric-vehicle-ev-battery-degradation-and-charge)
- 10,000 rows, 13 columns
- Target: `SoH_Percent` (range 73.74–100)
- Features: car model (5 models, roughly 2,000 vehicles each), battery type (NMC / LFP),
  battery capacity, vehicle age, total charging cycles, average temperature,
  fast charge ratio, average discharge rate, driving style, internal resistance

## Preprocessing
| Column | Encoding | Reason |
|---|---|---|
| `Car_Model` | One-Hot | Nominal, only 5 categories |
| `Battery_Type` | One-Hot | Nominal, 2 categories |
| `Driving_Style` | Ordinal (Conservative < Moderate < Aggressive) | Natural order |

Dropped from the features:
- `Vehicle_ID`: identifier, carries no information.
- `Battery_Status`: derived from the target, would cause data leakage.

## Why no classification model?
`Battery_Status` is extremely imbalanced (9,996 "Healthy" vs. 4 "Replace Required").
A model that always predicts "Healthy" would reach 99.96% accuracy while being useless.
In addition, the status is a threshold on `SoH_Percent` (about 80%: all "Healthy"
rows are ≥ 80.17, all "Replace Required" rows are ≤ 78.96), so a classifier would
only learn that threshold. Oversampling (e.g. SMOTE) cannot create real patterns
from 4 samples, so the project focuses on regression.

## Models and results
Train/test split: 80/20, `random_state=42`. Metrics are computed on the test set.

| Model | RMSE (test) | R² (test) |
|---|---|---|
| Linear Regression | 0.4132 | 0.9839 |
| Random Forest Regressor | 0.3770 | 0.9866 |

The Random Forest performs slightly better. The gap is small, which suggests that
the relationship between the features and SoH is mostly linear.

### Feature importance
`Internal_Resistance_Ohm` dominates the Random Forest (~95% of the impurity-based
importance, confirmed by permutation importance on the test set), followed by
`Total_Charging_Cycles` (~3%). All other features, including car model, driving
style and vehicle age, contribute almost nothing. Internal resistance is itself a
direct physical indicator of battery degradation, so it acts as a proxy for SoH
rather than an independent cause.

## Limitations
- The dataset appears to be synthetic, so results may not transfer to real-world data.
- Results come from a single train/test split; cross-validation would give a more
  reliable comparison between the two models.
- Impurity-based feature importance is biased towards continuous features and
  splits importance between correlated features.

## How to run
```bash
pip install pandas scikit-learn matplotlib
```
1. Download the dataset from Kaggle and place the CSV next to the notebook.
2. Update the file path in the first cell of the notebook if needed.
3. Open the notebook (e.g. in Google Colab) and run all cells.
