-- ═══════════════════════════════════════════════════════════════════
-- Urban Mobility Demand Forecasting 2024 — SQL Schema & Queries
-- Database: PostgreSQL
-- ═══════════════════════════════════════════════════════════════════


-- ─────────────────────────────────────────────────────────────────
-- 1. CREATE TABLES
-- ─────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS zones (
    zone_id       VARCHAR(15)  PRIMARY KEY,
    zone_name     VARCHAR(100) NOT NULL,
    city_area     VARCHAR(50),          -- e.g., 'Downtown', 'Suburbs'
    latitude      DECIMAL(9,6),
    longitude     DECIMAL(9,6),
    capacity      INT DEFAULT 100       -- max vehicles allocated
);

CREATE TABLE IF NOT EXISTS weather_log (
    weather_id    SERIAL PRIMARY KEY,
    recorded_at   TIMESTAMP NOT NULL,
    temperature   DECIMAL(5,2),
    rain          SMALLINT  DEFAULT 0,  -- 0 = dry, 1 = raining
    wind_speed    DECIMAL(5,2),
    CONSTRAINT chk_rain CHECK (rain IN (0,1))
);

CREATE TABLE IF NOT EXISTS mobility_events (
    event_id      SERIAL PRIMARY KEY,
    zone_id       VARCHAR(15) REFERENCES zones(zone_id),
    recorded_at   TIMESTAMP   NOT NULL,
    pickup_count  INT         NOT NULL CHECK (pickup_count >= 0),
    dropoff_count INT         NOT NULL CHECK (dropoff_count >= 0),
    traffic_level VARCHAR(10)          CHECK (traffic_level IN ('low','medium','high')),
    event_flag    SMALLINT    DEFAULT 0 CHECK (event_flag IN (0,1)),
    weather_id    INT REFERENCES weather_log(weather_id),
    hour_of_day   SMALLINT    GENERATED ALWAYS AS (EXTRACT(HOUR FROM recorded_at)) STORED,
    day_of_week   SMALLINT    GENERATED ALWAYS AS (EXTRACT(DOW  FROM recorded_at)) STORED
);

CREATE TABLE IF NOT EXISTS demand_predictions (
    prediction_id      SERIAL PRIMARY KEY,
    zone_id            VARCHAR(15) REFERENCES zones(zone_id),
    predicted_for      TIMESTAMP   NOT NULL,
    demand_label       VARCHAR(10) CHECK (demand_label IN ('LOW','MEDIUM','HIGH')),
    confidence_score   DECIMAL(5,4),
    model_version      VARCHAR(20) DEFAULT 'rf_v1.0',
    created_at         TIMESTAMP   DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_mobility_zone     ON mobility_events(zone_id);
CREATE INDEX IF NOT EXISTS idx_mobility_datetime ON mobility_events(recorded_at);
CREATE INDEX IF NOT EXISTS idx_mobility_hour     ON mobility_events(hour_of_day);
CREATE INDEX IF NOT EXISTS idx_pred_zone         ON demand_predictions(zone_id);
CREATE INDEX IF NOT EXISTS idx_pred_datetime     ON demand_predictions(predicted_for);


-- ─────────────────────────────────────────────────────────────────
-- 2. SEED DATA — Zones
-- ─────────────────────────────────────────────────────────────────
INSERT INTO zones (zone_id, zone_name, city_area, latitude, longitude, capacity) VALUES
  ('Zone_01', 'Central Business District', 'Downtown',  28.6139, 77.2090, 200),
  ('Zone_02', 'Airport Terminal',          'Airport',   28.5562, 77.1000, 300),
  ('Zone_03', 'Railway Station Hub',       'Transit',   28.6432, 77.2194, 250),
  ('Zone_04', 'University Campus',         'Suburbs',   28.7041, 77.1025, 120),
  ('Zone_05', 'Shopping District',         'Commercial',28.6289, 77.2310, 180),
  ('Zone_06', 'IT Park East',              'Tech Zone', 28.5921, 77.3200, 160),
  ('Zone_07', 'Residential North',         'Suburbs',   28.7500, 77.1800, 100),
  ('Zone_08', 'Hospital Complex',          'Medical',   28.6358, 77.2211, 140),
  ('Zone_09', 'Stadium & Events',          'Sports',    28.6100, 77.0900, 350),
  ('Zone_10', 'Old City Market',           'Heritage',  28.6553, 77.2310, 110),
  ('Zone_11', 'Financial District',        'Downtown',  28.6200, 77.2150, 190),
  ('Zone_12', 'Industrial Estate',         'Industrial',28.7050, 77.3100, 130),
  ('Zone_13', 'Convention Center',         'Commercial',28.6250, 77.2500, 220),
  ('Zone_14', 'Metro Interchange',         'Transit',   28.6400, 77.2100, 260),
  ('Zone_15', 'Residential South',         'Suburbs',   28.5300, 77.2200, 100),
  ('Zone_16', 'Tech Park West',            'Tech Zone', 28.5950, 77.0600, 160),
  ('Zone_17', 'Civic Center',              'Government',28.6150, 77.2050, 150),
  ('Zone_18', 'Entertainment Quarter',     'Commercial',28.5670, 77.2500, 210),
  ('Zone_19', 'Outer Ring Terminal',       'Transit',   28.7100, 77.1500, 180),
  ('Zone_20', 'Logistics Hub',             'Industrial',28.7400, 77.2800, 170)
ON CONFLICT (zone_id) DO NOTHING;


-- ─────────────────────────────────────────────────────────────────
-- 3. ANALYTICAL QUERIES
-- ─────────────────────────────────────────────────────────────────

-- ── Q1: Top 10 Highest Demand Zones (All Time) ──────────────────
SELECT
    z.zone_id,
    z.zone_name,
    z.city_area,
    COUNT(*)                                   AS total_records,
    ROUND(AVG(me.pickup_count), 1)             AS avg_pickups,
    ROUND(AVG(me.dropoff_count), 1)            AS avg_dropoffs,
    ROUND(AVG(me.pickup_count + me.dropoff_count), 1) AS avg_demand_intensity,
    SUM(me.pickup_count)                       AS total_pickups
FROM mobility_events me
JOIN zones z ON me.zone_id = z.zone_id
GROUP BY z.zone_id, z.zone_name, z.city_area
ORDER BY avg_demand_intensity DESC
LIMIT 10;


-- ── Q2: Peak Hour Demand Analysis ───────────────────────────────
SELECT
    hour_of_day,
    CASE
        WHEN hour_of_day BETWEEN 7  AND 9  THEN 'Morning Rush'
        WHEN hour_of_day BETWEEN 17 AND 19 THEN 'Evening Rush'
        WHEN hour_of_day BETWEEN 12 AND 14 THEN 'Lunch Hour'
        WHEN hour_of_day BETWEEN 22 AND 23 THEN 'Late Night'
        WHEN hour_of_day BETWEEN 0  AND 5  THEN 'Off Hours'
        ELSE 'Normal Hours'
    END                                         AS time_category,
    COUNT(*)                                    AS records,
    ROUND(AVG(pickup_count), 1)                 AS avg_pickups,
    ROUND(AVG(dropoff_count), 1)                AS avg_dropoffs,
    ROUND(AVG(pickup_count + dropoff_count), 1) AS avg_total_demand
FROM mobility_events
GROUP BY hour_of_day
ORDER BY avg_total_demand DESC;


-- ── Q3: Zone Comparison — Weekday vs Weekend ─────────────────────
SELECT
    z.zone_name,
    ROUND(AVG(CASE WHEN me.day_of_week BETWEEN 1 AND 5
                   THEN me.pickup_count END), 1)  AS avg_weekday_pickups,
    ROUND(AVG(CASE WHEN me.day_of_week IN (0, 6)
                   THEN me.pickup_count END), 1)  AS avg_weekend_pickups,
    ROUND(
        (AVG(CASE WHEN me.day_of_week IN (0,6)
                  THEN me.pickup_count END) -
         AVG(CASE WHEN me.day_of_week BETWEEN 1 AND 5
                  THEN me.pickup_count END))
        / NULLIF(AVG(CASE WHEN me.day_of_week BETWEEN 1 AND 5
                          THEN me.pickup_count END), 0) * 100, 1
    )                                             AS weekend_vs_weekday_pct
FROM mobility_events me
JOIN zones z ON me.zone_id = z.zone_id
GROUP BY z.zone_name
ORDER BY avg_weekday_pickups DESC;


-- ── Q4: High Demand Forecast — Upcoming Hours ────────────────────
SELECT
    dp.zone_id,
    z.zone_name,
    z.city_area,
    dp.predicted_for,
    dp.demand_label,
    ROUND(dp.confidence_score * 100, 1) AS confidence_pct,
    z.capacity                          AS zone_vehicle_capacity,
    CASE
        WHEN dp.demand_label = 'HIGH'   THEN ROUND(z.capacity * 0.90)
        WHEN dp.demand_label = 'MEDIUM' THEN ROUND(z.capacity * 0.60)
        ELSE                                 ROUND(z.capacity * 0.30)
    END                                 AS recommended_vehicles
FROM demand_predictions dp
JOIN zones z ON dp.zone_id = z.zone_id
WHERE dp.demand_label = 'HIGH'
  AND dp.predicted_for >= CURRENT_TIMESTAMP
ORDER BY dp.confidence_score DESC, dp.predicted_for ASC
LIMIT 20;


-- ── Q5: Traffic Level Impact on Demand ──────────────────────────
SELECT
    traffic_level,
    COUNT(*)                                    AS records,
    ROUND(AVG(pickup_count), 2)                 AS avg_pickups,
    ROUND(STDDEV(pickup_count), 2)              AS stddev_pickups,
    ROUND(AVG(pickup_count + dropoff_count), 2) AS avg_total_demand,
    ROUND(MAX(pickup_count + dropoff_count), 2) AS max_demand_seen
FROM mobility_events
WHERE traffic_level IS NOT NULL
GROUP BY traffic_level
ORDER BY avg_total_demand DESC;


-- ── Q6: Monthly Demand Trend ─────────────────────────────────────
SELECT
    DATE_TRUNC('month', recorded_at) AS month,
    COUNT(*)                          AS total_records,
    ROUND(SUM(pickup_count), 0)       AS total_pickups,
    ROUND(AVG(pickup_count), 1)       AS avg_pickups,
    ROUND(AVG(pickup_count + dropoff_count), 1) AS avg_demand
FROM mobility_events
GROUP BY DATE_TRUNC('month', recorded_at)
ORDER BY month;


-- ── Q7: Event Impact Analysis ────────────────────────────────────
SELECT
    event_flag,
    CASE event_flag WHEN 1 THEN 'Special Event' ELSE 'Normal Day' END AS day_type,
    COUNT(*)                               AS records,
    ROUND(AVG(pickup_count), 1)            AS avg_pickups,
    ROUND(AVG(dropoff_count), 1)           AS avg_dropoffs,
    ROUND(AVG(pickup_count + dropoff_count), 1) AS avg_demand
FROM mobility_events
GROUP BY event_flag
ORDER BY event_flag;


-- ── Q8: Zone Resource Allocation Recommendation ──────────────────
WITH zone_stats AS (
    SELECT
        zone_id,
        ROUND(AVG(pickup_count + dropoff_count), 1)  AS avg_demand,
        ROUND(STDDEV(pickup_count + dropoff_count), 1) AS demand_volatility,
        ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP
              (ORDER BY pickup_count + dropoff_count), 0) AS p95_demand
    FROM mobility_events
    GROUP BY zone_id
)
SELECT
    z.zone_id,
    z.zone_name,
    z.capacity                   AS current_capacity,
    zs.avg_demand,
    zs.p95_demand,
    zs.demand_volatility,
    CASE
        WHEN zs.p95_demand > z.capacity * 0.9 THEN 'INCREASE FLEET'
        WHEN zs.avg_demand  < z.capacity * 0.3 THEN 'REDUCE FLEET'
        ELSE 'MAINTAIN'
    END                          AS recommendation
FROM zone_stats zs
JOIN zones z ON zs.zone_id = z.zone_id
ORDER BY zs.avg_demand DESC;
