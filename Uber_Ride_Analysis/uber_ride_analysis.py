"""
Uber Ride Analysis and Hourly Demand Forecasting
------------------------------------------------
Run:
    pip install -r requirements.txt
    python uber_ride_analysis.py --data uber.csv

The script:
- Loads and validates a ride dataset
- Cleans timestamps and removes invalid/duplicate rows
- Performs exploratory data analysis
- Saves charts and summary files to outputs/
- Optionally analyzes fares if a real fare column exists
- Trains and evaluates an hourly ride-demand forecasting model

Expected minimum column:
    pickup_datetime (or a recognized equivalent such as pickup_date)

Optional columns:
    fare_amount, pickup_longitude, pickup_latitude, etc.
"""

from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

TIME_CANDIDATES = [
    "pickup_datetime", "pickup_date", "datetime", "timestamp",
    "date_time", "date/time", "tpep_pickup_datetime",
    "lpep_pickup_datetime"
]
FARE_CANDIDATES = ["fare_amount", "fare", "trip_fare", "total_amount"]


def find_column(columns, candidates):
    """Find a column case-insensitively, allowing surrounding spaces."""
    lookup = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def save_plot(filename):
    """Save the current Matplotlib figure and close it."""
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close()


def main(data_path):
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Put the CSV beside this script "
            "or pass its path using --data."
        )

    print("\n[1/6] Loading dataset...")
    df = pd.read_csv(
    data_path,
    encoding="latin1",
    engine="python",
    on_bad_lines="warn"
)
    df.columns = [str(c).strip() for c in df.columns]
    print(f"Original shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("Columns:", list(df.columns))

    time_col = find_column(df.columns, TIME_CANDIDATES)
    if time_col is None:
        # Try likely date/time column names if the exact candidates differ.
        possible = [
            c for c in df.columns
            if ("pickup" in c.lower() and ("time" in c.lower() or "date" in c.lower()))
            or ("datetime" in c.lower())
        ]
        if possible:
            time_col = possible[0]
        else:
            raise ValueError(
                "Could not identify the pickup timestamp column. Rename your "
                "date/time column to 'pickup_datetime' and try again."
            )

    print("\n[2/6] Preprocessing...")
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[time_col]).drop_duplicates().copy()
    df = df.sort_values(time_col).reset_index(drop=True)
    print(f"Removed {before - len(df):,} rows with invalid timestamps or duplicates.")
    if df.empty:
        raise ValueError("No valid rows remain after timestamp cleaning.")

    # Standardized time features
    df["pickup_datetime_clean"] = df[time_col]
    df["date"] = df["pickup_datetime_clean"].dt.date
    df["year"] = df["pickup_datetime_clean"].dt.year
    df["month"] = df["pickup_datetime_clean"].dt.month
    df["day"] = df["pickup_datetime_clean"].dt.day
    df["day_of_week"] = df["pickup_datetime_clean"].dt.dayofweek
    df["day_name"] = df["pickup_datetime_clean"].dt.day_name()
    df["hour"] = df["pickup_datetime_clean"].dt.hour
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Save cleaned data
    df.to_csv(OUTPUT_DIR / "cleaned_uber_data.csv", index=False)
    df.describe(include="all").transpose().to_csv(OUTPUT_DIR / "data_summary.csv")

    print("\n[3/6] Exploratory data analysis and charts...")
    # Rides by hour
    hourly_counts = df.groupby("hour").size().reindex(range(24), fill_value=0)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=hourly_counts.index, y=hourly_counts.values, color="steelblue")
    plt.title("Uber Pickups by Hour of Day")
    plt.xlabel("Hour (0–23)")
    plt.ylabel("Number of pickups")
    save_plot("rides_by_hour.png")

    # Rides by weekday
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_counts = df["day_name"].value_counts().reindex(weekday_order, fill_value=0)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=weekday_counts.index, y=weekday_counts.values, color="darkcyan")
    plt.title("Uber Pickups by Day of Week")
    plt.xlabel("Day")
    plt.ylabel("Number of pickups")
    plt.xticks(rotation=25)
    save_plot("rides_by_weekday.png")

    # Date-based daily trend
    daily_counts = df.set_index("pickup_datetime_clean").resample("D").size()
    plt.figure(figsize=(12, 5))
    plt.plot(daily_counts.index, daily_counts.values, linewidth=1.5)
    plt.title("Daily Uber Pickup Trend")
    plt.xlabel("Date")
    plt.ylabel("Number of pickups")
    save_plot("daily_ride_trend.png")

    # Hour x weekday heatmap
    heatmap_data = df.pivot_table(
        index="day_name", columns="hour", values=time_col,
        aggfunc="count", fill_value=0
    ).reindex(weekday_order)
    plt.figure(figsize=(13, 5))
    sns.heatmap(heatmap_data, cmap="YlGnBu", linewidths=0.2)
    plt.title("Pickup Demand Heatmap: Day of Week vs Hour")
    plt.xlabel("Hour (0–23)")
    plt.ylabel("Day of week")
    save_plot("demand_heatmap.png")

    # Fare analysis only when a real fare column exists
    fare_col = find_column(df.columns, FARE_CANDIDATES)
    fare_insights = []
    if fare_col is not None:
        df[fare_col] = pd.to_numeric(df[fare_col], errors="coerce")
        valid_fares = df.loc[df[fare_col].notna() & (df[fare_col] >= 0)].copy()
        if not valid_fares.empty:
            plt.figure(figsize=(9, 5))
            sns.histplot(valid_fares[fare_col], bins=40, kde=True)
            plt.title(f"Distribution of {fare_col}")
            plt.xlabel("Fare amount")
            plt.ylabel("Number of rides")
            save_plot("fare_distribution.png")

            fare_by_hour = valid_fares.groupby("hour")[fare_col].median()
            plt.figure(figsize=(10, 5))
            sns.lineplot(x=fare_by_hour.index, y=fare_by_hour.values, marker="o")
            plt.title(f"Median {fare_col} by Pickup Hour")
            plt.xlabel("Hour (0–23)")
            plt.ylabel("Median fare")
            save_plot("fare_trend_by_hour.png")

            fare_insights = [
                f"Fare column used: {fare_col}",
                f"Valid non-negative fare records: {len(valid_fares):,}",
                f"Mean fare: {valid_fares[fare_col].mean():.2f}",
                f"Median fare: {valid_fares[fare_col].median():.2f}",
            ]
        else:
            fare_insights = [f"Fare column '{fare_col}' exists, but no valid non-negative numeric fares were found."]
    else:
        fare_insights = [
            "Fare analysis skipped: this dataset has no recognized fare column.",
            "Do not infer fare trends from pickup-only data. Use a trip/fare dataset to analyze pricing."
        ]

    # Location plot if longitude and latitude fields are available
    lon_col = find_column(df.columns, ["pickup_longitude", "longitude", "pickup_lon"])
    lat_col = find_column(df.columns, ["pickup_latitude", "latitude", "pickup_lat"])
    if lon_col and lat_col:
        locations = df[[lon_col, lat_col]].apply(pd.to_numeric, errors="coerce").dropna()
        # Filter obviously invalid coordinates for readable plots
        locations = locations[
            locations[lon_col].between(-180, 180) &
            locations[lat_col].between(-90, 90)
        ]
        if len(locations) > 10000:
            locations = locations.sample(10000, random_state=42)
        if not locations.empty:
            plt.figure(figsize=(8, 7))
            plt.scatter(locations[lon_col], locations[lat_col], s=3, alpha=0.25)
            plt.title("Pickup Locations (sampled when dataset is large)")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            save_plot("pickup_locations.png")

    print("\n[4/6] Building hourly demand forecasting model...")
    # Aggregate into a continuous hourly series, including zero-ride hours.
    hourly = df.set_index("pickup_datetime_clean").resample("h").size().rename("ride_count")
    if len(hourly) < 72:
        model_message = (
            "Forecasting skipped: at least 72 hourly periods are recommended. "
            "Use a larger date range for a more meaningful time-series model."
        )
        model_metrics = None
    else:
        features = pd.DataFrame(index=hourly.index)
        features["hour"] = hourly.index.hour
        features["day_of_week"] = hourly.index.dayofweek
        features["day_of_month"] = hourly.index.day
        features["month"] = hourly.index.month
        features["is_weekend"] = (hourly.index.dayofweek >= 5).astype(int)
        features["lag_1"] = hourly.shift(1)
        features["lag_24"] = hourly.shift(24)
        features["lag_168"] = hourly.shift(168)
        features["rolling_mean_24"] = hourly.shift(1).rolling(24).mean()
        features["target"] = hourly
        model_data = features.dropna()

        if len(model_data) < 48:
            model_message = "Forecasting skipped: not enough rows after creating lag features."
            model_metrics = None
        else:
            split = int(len(model_data) * 0.8)
            train = model_data.iloc[:split]
            test = model_data.iloc[split:]
            X_train = train.drop(columns="target")
            y_train = train["target"]
            X_test = test.drop(columns="target")
            y_test = test["target"]

            model = RandomForestRegressor(
                n_estimators=250, random_state=42, min_samples_leaf=2,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            predictions = np.maximum(0, model.predict(X_test))
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions) if len(y_test) > 1 else float("nan")
            model_metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
            model_message = "Random Forest trained using a chronological 80/20 train-test split."

            plt.figure(figsize=(12, 5))
            plt.plot(y_test.index, y_test.values, label="Actual rides", linewidth=1.5)
            plt.plot(y_test.index, predictions, label="Predicted rides", linewidth=1.5)
            plt.title("Hourly Ride Demand: Actual vs Predicted")
            plt.xlabel("Time")
            plt.ylabel("Ride count")
            plt.legend()
            save_plot("demand_forecast.png")

            importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values()
            plt.figure(figsize=(8, 5))
            importance.plot(kind="barh")
            plt.title("Demand Model Feature Importance")
            plt.xlabel("Importance")
            save_plot("model_feature_importance.png")

            pd.DataFrame({
                "timestamp": y_test.index,
                "actual_rides": y_test.values,
                "predicted_rides": predictions
            }).to_csv(OUTPUT_DIR / "forecast_test_predictions.csv", index=False)

    print("\n[5/6] Generating insights and report...")
    peak_hour = int(hourly_counts.idxmax())
    quiet_hour = int(hourly_counts.idxmin())
    peak_day = str(weekday_counts.idxmax())
    quiet_day = str(weekday_counts.idxmin())

    insights = [
        "UBER RIDE ANALYSIS — AUTOMATIC SUMMARY",
        "=" * 42,
        f"Source file: {data_path.name}",
        f"Cleaned rides: {len(df):,}",
        f"Date range: {df['pickup_datetime_clean'].min()} to {df['pickup_datetime_clean'].max()}",
        f"Unique dates: {df['date'].nunique():,}",
        f"Peak pickup hour in this dataset: {peak_hour}:00 ({int(hourly_counts.max()):,} rides)",
        f"Lowest pickup hour in this dataset: {quiet_hour}:00 ({int(hourly_counts.min()):,} rides)",
        f"Busiest weekday: {peak_day} ({int(weekday_counts.max()):,} rides)",
        f"Least busy weekday: {quiet_day} ({int(weekday_counts.min()):,} rides)",
        "",
        "FARE ANALYSIS",
        *fare_insights,
        "",
        "DEMAND FORECASTING",
        model_message,
    ]
    if model_metrics:
        insights += [
            f"MAE: {model_metrics['MAE']:.3f} rides",
            f"RMSE: {model_metrics['RMSE']:.3f} rides",
            f"R²: {model_metrics['R2']:.3f}",
            "Metrics are evaluated on the final 20% of the timeline; performance depends on dataset size and quality."
        ]

    insights += [
        "",
        "INTERPRETATION NOTES",
        "- These findings describe the supplied dataset and its date/location coverage only.",
        "- Correlation or a busy hour does not by itself establish the cause of demand.",
        "- A pickup-only dataset cannot support genuine fare or surge-price conclusions.",
        "- For stronger forecasting, compare with seasonal-naive baselines and validate on later dates.",
        "- Potential uses: driver scheduling, fleet allocation, service coverage, and planning for busy periods."
    ]
    (OUTPUT_DIR / "key_insights.txt").write_text("\n".join(insights), encoding="utf-8")

    if model_metrics:
        pd.DataFrame([model_metrics]).to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    print("\n".join(insights))
    print("\n[6/6] Finished!")
    print(f"Charts, cleaned data, and results saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Uber ride pickups and forecast hourly demand.")
    parser.add_argument("--data", default="uber.csv", help="Path to the input CSV dataset (default: uber.csv)")
    args = parser.parse_args()
    main(args.data)
