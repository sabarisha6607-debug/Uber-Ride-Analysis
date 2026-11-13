# Uber Ride Analysis Using Data Science

## Project overview
This project analyzes Uber pickup records using Python data-science libraries. It explores ride demand by hour and weekday, daily trends, demand heatmaps, optional fare trends, and hourly demand forecasting with a Random Forest regressor.

## Dataset
Start with the [FiveThirtyEight Uber NYC pickups dataset on Kaggle](https://www.kaggle.com/datasets/fivethirtyeight/uber-pickups-in-new-york-city). Download a CSV file and place it in this project folder as `uber.csv`, or pass its path when running the script.

The script needs a pickup timestamp column. It recognizes common names such as `pickup_datetime`, `pickup_date`, `datetime`, and `timestamp`. Fare analysis runs only when the CSV contains a numeric fare-like column such as `fare_amount` or `total_amount`.

**Dataset limitation:** Pickup-only files do not contain actual fare values. In that case, the program skips fare analysis rather than creating misleading prices. Use a trip-level dataset with genuine fare information to study pricing.

## Installation
Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install packages:
```bash
pip install -r requirements.txt
```

## Run the project
Put `uber.csv` in this folder and run:

```bash
python uber_ride_analysis.py --data uber.csv
```

Or use another filename/path:
```bash
python uber_ride_analysis.py --data "data/uber_pickups.csv"
```

## Preprocessing steps
1. Read the CSV and standardize column-name whitespace.
2. Detect the pickup timestamp field.
3. Parse timestamps and remove records with invalid timestamps.
4. Remove duplicate rows and sort chronologically.
5. Derive hour, weekday, month, date, and weekend features.
6. Save cleaned records and summary statistics.

## Exploratory data analysis
The program creates:
- `rides_by_hour.png` — pickup count by hour.
- `rides_by_weekday.png` — pickup count by weekday.
- `daily_ride_trend.png` — daily pickup trend.
- `demand_heatmap.png` — weekday-by-hour demand heatmap.
- `fare_distribution.png` and `fare_trend_by_hour.png` — only if a valid fare column exists.
- `pickup_locations.png` — only if latitude and longitude columns exist.

## Forecasting approach
The script aggregates rides into continuous hourly counts and trains a `RandomForestRegressor`. Features include hour, weekday, month, weekend flag, previous-hour demand, demand 24 hours earlier, demand 168 hours earlier, and a rolling 24-hour average. The first 80% of the timeline is used for training and the last 20% for testing to avoid random time leakage.

Metrics:
- **MAE:** average absolute prediction error, in rides.
- **RMSE:** gives larger errors more weight, in rides.
- **R²:** compares predictions with a mean-baseline reference; it may be negative if the model performs poorly.

The model is an educational baseline, not a guaranteed production forecasting system. For rigorous evaluation, compare it with a seasonal-naive baseline and use rolling time-series validation.

## Output files
All generated results are saved to `outputs/`:
- `cleaned_uber_data.csv`
- `data_summary.csv`
- `key_insights.txt`
- `model_metrics.csv` (when a model is trained)
- `forecast_test_predictions.csv` (when a model is trained)
- PNG charts listed above

## How to write the findings
Run the program first, then use the actual figures in `outputs/key_insights.txt` and the generated charts. Do not write assumed peak hours or fares before analyzing your selected file: results depend on the dataset and its time/location coverage.

## Challenges and business applications
Possible challenges include missing timestamps, inconsistent schemas, incomplete date coverage, unusual coordinate values, and changes in demand over time. Insights can support driver scheduling, fleet allocation, service availability, and planning for busy periods. Fare optimization requires a dataset with genuine fare and ideally trip-distance, duration, and surge information.

## Suggested GitHub repository contents
```text
Uber_Ride_Analysis/
├── uber_ride_analysis.py
├── requirements.txt
├── README.md
├── .gitignore
├── uber.csv                 # optional; respect the dataset license and size
└── outputs/                 # generated results
```
