# Urban Mobility Demand Forecasting 2024

Scenario-based urban mobility demand classification project using synthetic transport data, a Random Forest classifier, generated charts, SQL analysis queries, and a Streamlit dashboard.

This is a portfolio/demo project. The 24-hour output is a **demand scenario forecast** generated from assumed future conditions, not a production-grade real-world forecasting system.

## Project Structure

```text
urban-mobility-forecasting/
├── app.py                         # Streamlit dashboard
├── main_pipeline.py               # Data generation, preprocessing, ML, scenario forecast
├── requirements.txt
├── README.md
├── RUN_INSTRUCTIONS.md
├── schema_and_queries.sql         # PostgreSQL schema + analytical queries
├── powerbi_guide.md
├── data/
│   ├── raw_mobility_data.csv
│   ├── processed_data.csv
│   ├── forecast_output.csv
│   └── top_zones_forecast.csv
├── notebooks/
│   ├── eda_charts.png
│   ├── model_evaluation.png
│   └── forecast_charts.png
└── models/
    └── random_forest.joblib
```

## Quick Start

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python main_pipeline.py
streamlit run app.py
```

For macOS/Linux, use `source venv/bin/activate` instead of `venv\Scripts\activate`.

## Dataset

The pipeline generates 10,000 hourly synthetic mobility records across `Zone_01` through `Zone_20`.

Key fields include:

- `datetime`, `zone_id`
- `pickup_count`, `dropoff_count`
- `temperature`, `rain`, `traffic_level`, `event_flag`
- `hour_of_day`, `day_of_week`, `month`, `quarter`
- `peak_hour_flag`, `weekend_flag`
- `demand_intensity = pickup_count + dropoff_count`
- `demand_label` target: `LOW`, `MEDIUM`, `HIGH`

## Model

The model is a Random Forest classifier trained to classify demand labels. To avoid target leakage, the ML feature list excludes current-demand-derived fields such as `demand_traffic_ratio` and `net_flow`.

Training features:

- `hour_of_day`
- `day_of_week`
- `month`
- `quarter`
- `temperature`
- `rain`
- `event_flag`
- `peak_hour_flag`
- `weekend_flag`
- `zone_encoded`
- `traffic_encoded`

Latest pipeline run metrics after leakage removal:

| Metric | Score |
| --- | ---: |
| Accuracy | 81.5% |
| Weighted precision | 0.8102 |
| Weighted recall | 0.8145 |
| Weighted F1 | 0.8083 |

The trained model is saved to `models/random_forest.joblib`.

## Scenario Forecast Output

The pipeline creates a 24-hour demand scenario for all 20 zones using assumed hour, traffic, weather, and event inputs. Outputs:

- `data/forecast_output.csv`
- `data/top_zones_forecast.csv`
- `notebooks/forecast_charts.png`

These files are useful for dashboard exploration and portfolio demonstration. They should not be described as measured production forecasts.

## SQL

`schema_and_queries.sql` is written for PostgreSQL. It uses PostgreSQL-specific features including `SERIAL`, generated columns, `DATE_TRUNC`, `STDDEV`, and `PERCENTILE_CONT`.

Run with:

```bash
psql -U your_user -d your_db -f schema_and_queries.sql
```

## Dashboard

Launch the dashboard with:

```bash
streamlit run app.py
```

Dashboard pages:

- Overview
- Demand Analysis
- Scenario Forecast
- Zone Explorer
- Model Performance

## Business Interpretation

The project can illustrate how demand classification might support fleet positioning, peak-hour planning, and zone-level monitoring. It does not calculate measured improvements in wait time, fleet utilization, revenue, or stakeholder decision speed.

## Resume-Friendly Summary

- Built an end-to-end synthetic urban mobility demand classification project using Python, pandas, scikit-learn, and Streamlit.
- Trained a leakage-aware Random Forest classifier with 81.5% accuracy and 0.8083 weighted F1 on 10,000 generated records.
- Produced a PostgreSQL schema, analytical SQL queries, reproducible data outputs, charts, and a 24-hour demand scenario dashboard.
