# How to Run Urban Mobility Demand Forecasting

## Prerequisites

- Python 3.10 or higher
- Windows, macOS, or Linux

## Setup

```bash
pip install -r requirements.txt
```

## Run the Pipeline

```bash
python main_pipeline.py
```

This generates or refreshes:

- `data/raw_mobility_data.csv`
- `data/processed_data.csv`
- `data/forecast_output.csv`
- `data/top_zones_forecast.csv`
- `notebooks/eda_charts.png`
- `notebooks/model_evaluation.png`
- `notebooks/forecast_charts.png`
- `models/random_forest.joblib`

The forecast files are 24-hour demand scenario outputs based on assumed future inputs.

## Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

Run `main_pipeline.py` first so the dashboard data files exist.

## Project Structure

```text
urban-mobility-forecasting/
├── app.py
├── main_pipeline.py
├── requirements.txt
├── README.md
├── RUN_INSTRUCTIONS.md
├── schema_and_queries.sql
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

## Dashboard Pages

1. Overview
2. Demand Analysis
3. Scenario Forecast
4. Zone Explorer
5. Model Performance

## SQL

The SQL file is PostgreSQL-specific:

```bash
psql -U your_user -d your_db -f schema_and_queries.sql
```

## Troubleshooting

Missing required data files in the dashboard:

```bash
python main_pipeline.py
```

Module not found:

```bash
pip install -r requirements.txt
```

Port 8501 already in use:

```bash
streamlit run app.py --server.port 8502
```
