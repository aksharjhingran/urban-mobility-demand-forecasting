# 📊 Power BI Dashboard — Setup Guide
## Urban Mobility Demand Forecasting 2024

---

## 🔌 Data Connection

1. Open Power BI Desktop → **Get Data → Text/CSV**
2. Load these files:
   - `data/processed_data.csv` (main fact table)
   - `data/forecast_output.csv` (prediction table)
   - `data/top_zones_forecast.csv` (ranking table)
3. Go to **Transform Data** → verify column types

---

## 📐 Data Model (Star Schema)

```
processed_data  ──┐
                  ├─► (Zone_id joins)
forecast_output ──┘
       │
       └── top_zones_forecast (summary)
```

Create relationships:
- `processed_data[zone_id]` → `forecast_output[zone_id]`
- Type: Many-to-Many | Cross-filter: Both

---

## 🧮 DAX Measures to Create

```dax
-- Total Pickups
Total Pickups = SUM(processed_data[pickup_count])

-- Average Demand Intensity
Avg Demand Intensity = AVERAGE(processed_data[demand_intensity])

-- Peak Hour Pickups
Peak Pickups = 
CALCULATE(
    SUM(processed_data[pickup_count]),
    processed_data[peak_hour_flag] = 1
)

-- High Demand Zone Count
High Demand Zones = 
CALCULATE(
    DISTINCTCOUNT(forecast_output[zone_id]),
    forecast_output[predicted_demand] = "HIGH"
)

-- Forecast Confidence (Avg)
Avg Confidence = 
AVERAGE(forecast_output[demand_proba])

-- Weekend vs Weekday ratio
Weekend Lift % = 
DIVIDE(
    CALCULATE(AVERAGE(processed_data[pickup_count]),
              processed_data[weekend_flag] = 1),
    CALCULATE(AVERAGE(processed_data[pickup_count]),
              processed_data[weekend_flag] = 0)
) - 1
```

---

## 📄 PAGE 1 — Overview Dashboard

### KPI Cards (top row):
| Card | Measure | Icon |
|------|---------|------|
| Total Pickups | `Total Pickups` | 🚖 |
| Avg Demand Intensity | `Avg Demand Intensity` | 📈 |
| Peak Hour Pickups | `Peak Pickups` | ⏱️ |
| High-Demand Zones (24h) | `High Demand Zones` | 📍 |

### Charts:
- **Line Chart** — Total Pickups by `datetime` (month granularity)
  - X: month, Y: Total Pickups, Legend: weekend_flag
- **Donut Chart** — demand_label distribution (LOW/MEDIUM/HIGH)
  - Colors: Green, Orange, Red
- **Bar Chart** — Top 10 zones by Avg Demand Intensity
  - Sorted descending, conditional color formatting

### Filters/Slicers (right panel):
- Date Range slicer (`datetime`)
- Zone multi-select (`zone_id`)
- Traffic Level (`traffic_level`)

---

## 📄 PAGE 2 — Demand Trends

### Charts:
- **Area Chart** — Hourly demand trend
  - X: hour_of_day (0–23), Y: Avg pickup_count
  - Add reference line at hour 8 and hour 18

- **Clustered Bar** — Day of Week vs Avg Pickups
  - X: day_of_week, Y: Avg pickup_count
  - Rename axis: 0=Mon, 1=Tue, … 6=Sun

- **Heatmap Matrix** (using Matrix visual)
  - Rows: hour_of_day
  - Columns: day_of_week
  - Values: Avg demand_intensity
  - Background color: gradient Green→Red

- **Line Chart** — Event vs Non-Event demand comparison
  - X: hour_of_day, Y: Avg pickups
  - Legend: event_flag (0 = Normal, 1 = Event)

### KPIs on page:
- Peak vs Off-Peak demand ratio
- Weekend Lift %

---

## 📄 PAGE 3 — High-Demand Zones (Map View)

> **Requires:** Add latitude/longitude to zone master data

### Visuals:
- **Filled Map** or **ArcGIS Map**
  - Location: zone_id
  - Bubble size: Avg demand_intensity
  - Color saturation: predicted_demand (HIGH = Red)

- **Table** — Zone Ranking
  | Column | Source |
  |--------|--------|
  | Zone | zone_id |
  | Predicted Demand | predicted_demand |
  | High-Demand Hours | high_demand_hours |
  | Confidence % | avg_confidence |
  | Recommended Action | (conditional column) |

- **Bar Chart** — Top zones by high_demand_hours
  - Color: ACCENT gradient

### Conditional Formatting Rule (Table):
- predicted_demand = "HIGH" → Red background
- predicted_demand = "MEDIUM" → Orange
- predicted_demand = "LOW" → Green

---

## 📄 PAGE 4 — Forecast Insights

### Visuals:
- **Stacked Bar** — Demand label distribution per zone (top 10)
  - X: zone_id, Y: count of records
  - Legend: demand_label (3 colors)

- **Gauge Chart** — Model Avg Confidence
  - Value: `Avg Confidence` measure
  - Target: 0.85 (85%)

- **Table** — Next 24h High Demand Zones
  | Column | Source |
  |--------|--------|
  | Zone | zone_id |
  | Hour | hour_of_day |
  | Predicted | predicted_demand |
  | Confidence | demand_proba |

- **Line + Column** — Hourly forecast across sample zones
  - X: hour_of_day, Y: Avg pickup_count
  - Line: avg confidence

### Insight Text Box (add manually):
> "Zones in the top 25% demand quantile are predicted HIGH for 
>  8+ hours tomorrow. Pre-deploy vehicles by 7:00 AM."

---

## 🎨 Dashboard Theme Settings

```json
{
  "name": "Urban Mobility Dark",
  "dataColors": ["#7c6fcd", "#56cfb2", "#f7c59f", "#e84393", "#60a5fa"],
  "background": "#0f1117",
  "foreground": "#e0e0e0",
  "tableAccent": "#7c6fcd",
  "visualStyles": {
    "*": { "*": { "fontFamily": [{ "value": "Segoe UI" }] } }
  }
}
```

Save as `.json` → Power BI Desktop → View → Themes → Browse for themes

---

## ✅ Publishing Checklist

- [ ] All 4 pages created
- [ ] DAX measures validated
- [ ] Slicers cross-filter all pages
- [ ] Mobile layout configured (View → Mobile Layout)
- [ ] Published to Power BI Service
- [ ] Row-Level Security (RLS) configured by zone (if needed)
- [ ] Scheduled refresh set (daily at 6 AM)
