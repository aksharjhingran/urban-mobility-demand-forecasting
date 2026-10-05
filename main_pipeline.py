"""
╔══════════════════════════════════════════════════════════════════╗
║      URBAN MOBILITY DEMAND FORECASTING 2024                      ║
║      Full ML Pipeline: Data → EDA → Model → Forecast             ║
║      Cross-platform (Windows / Mac / Linux)                      ║
╚══════════════════════════════════════════════════════════════════╝

USAGE:
    Just put this file anywhere and run:
        python main_pipeline.py

    Folders (data/, notebooks/) will be created automatically
    in the SAME directory as this script.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CROSS-PLATFORM PATH SETUP — works on Windows / Mac / Linux
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCRIPT_DIR    = Path(__file__).resolve().parent
DATA_DIR      = SCRIPT_DIR / 'data'
NOTEBOOKS_DIR = SCRIPT_DIR / 'notebooks'
MODELS_DIR    = SCRIPT_DIR / 'models'

# Auto-create folders if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# File paths
RAW_CSV       = DATA_DIR / 'raw_mobility_data.csv'
PROCESSED_CSV = DATA_DIR / 'processed_data.csv'
FORECAST_CSV  = DATA_DIR / 'forecast_output.csv'
TOP_ZONES_CSV = DATA_DIR / 'top_zones_forecast.csv'
EDA_PNG       = NOTEBOOKS_DIR / 'eda_charts.png'
MODEL_PNG     = NOTEBOOKS_DIR / 'model_evaluation.png'
FORECAST_PNG  = NOTEBOOKS_DIR / 'forecast_charts.png'
MODEL_FILE    = MODELS_DIR / 'random_forest.joblib'

np.random.seed(42)
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor': '#1a1d2e',
    'axes.edgecolor': '#2d3250',
    'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#a0a0b0',
    'ytick.color': '#a0a0b0',
    'text.color': '#e0e0e0',
    'grid.color': '#2d3250',
    'grid.alpha': 0.5,
    'figure.dpi': 120,
})

ACCENT  = '#7c6fcd'
ACCENT2 = '#56cfb2'
ACCENT3 = '#f7c59f'
ACCENT4 = '#e84393'
PALETTE = [ACCENT, ACCENT2, ACCENT3, ACCENT4, '#60a5fa', '#fb923c']

print("=" * 65)
print("  URBAN MOBILITY DEMAND FORECASTING 2024")
print("  Building Complete ML Pipeline...")
print("=" * 65)
print(f"  Working directory: {SCRIPT_DIR}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2: SYNTHETIC DATASET GENERATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[STEP 2] Generating Synthetic Dataset...")

N_RECORDS = 10_000
ZONES = [f"Zone_{i:02d}" for i in range(1, 21)]

date_range = pd.date_range(start='2024-01-01', periods=N_RECORDS, freq='h')

def simulate_pickups(hour, dow, zone_id, event, traffic, temp):
    base = np.random.poisson(30)
    if 7 <= hour <= 9:   base += np.random.randint(40, 80)
    if 17 <= hour <= 19: base += np.random.randint(50, 90)
    if 0 <= hour <= 5:   base = max(1, base - 20)
    if dow >= 5 and 22 <= hour: base += np.random.randint(20, 50)
    if event: base += np.random.randint(60, 120)
    traffic_map = {'low': -5, 'medium': 0, 'high': 15}
    base += traffic_map.get(traffic, 0)
    if temp < 5:  base -= 10
    if temp > 35: base -= 8
    return max(1, base + np.random.randint(-5, 5))

hours      = date_range.hour.values
dow        = date_range.dayofweek.values
zones      = np.random.choice(ZONES, N_RECORDS)
temp       = np.random.normal(22, 8, N_RECORDS).clip(0, 45).round(1)
rain       = np.random.choice([0, 0, 0, 1], N_RECORDS)
traffic    = np.random.choice(['low','medium','high'], N_RECORDS,
                               p=[0.35, 0.40, 0.25])
event_flag = np.random.choice([0, 0, 0, 0, 1], N_RECORDS)

pickups  = [simulate_pickups(h, d, z, e, t, tmp)
            for h, d, z, e, t, tmp in
            zip(hours, dow, zones, event_flag, traffic, temp)]
dropoffs = [max(1, p + np.random.randint(-10, 10)) for p in pickups]

df = pd.DataFrame({
    'datetime'    : date_range,
    'zone_id'     : zones,
    'pickup_count': pickups,
    'dropoff_count': dropoffs,
    'temperature' : temp,
    'rain'        : rain,
    'day_of_week' : dow,
    'hour_of_day' : hours,
    'traffic_level': traffic,
    'event_flag'  : event_flag,
})

# Introduce ~2% missing values (realistic data quality)
for col in ['pi'
'ckup_count', 'temperature', 'traffic_level']:
    mask = np.random.rand(N_RECORDS) < 0.02
    df.loc[mask, col] = np.nan

df.to_csv(RAW_CSV, index=False)
print(f"  ✓ Dataset shape: {df.shape}")
print(f"  ✓ Date range  : {df['datetime'].min()} → {df['datetime'].max()}")
print(f"  ✓ Zones       : {df['zone_id'].nunique()} unique zones")
print(f"  ✓ Missing vals: {df.isnull().sum().sum()} cells")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3: PREPROCESSING & FEATURE ENGINEERING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[STEP 3] Preprocessing & Feature Engineering...")

df_clean = df.copy()

# Handle missing values
df_clean['pickup_count'].fillna(df_clean['pickup_count'].median(), inplace=True)
df_clean['temperature'].fillna(df_clean['temperature'].mean(), inplace=True)
df_clean['traffic_level'].fillna('medium', inplace=True)

# DateTime feature extraction
df_clean['datetime'] = pd.to_datetime(df_clean['datetime'])
df_clean['month']    = df_clean['datetime'].dt.month
df_clean['week']     = df_clean['datetime'].dt.isocalendar().week.astype(int)
df_clean['quarter']  = df_clean['datetime'].dt.quarter

# Feature Engineering
df_clean['peak_hour_flag'] = df_clean['hour_of_day'].apply(
    lambda h: 1 if (7 <= h <= 9) or (17 <= h <= 19) else 0
)
df_clean['weekend_flag']     = (df_clean['day_of_week'] >= 5).astype(int)
df_clean['demand_intensity'] = df_clean['pickup_count'] + df_clean['dropoff_count']
df_clean['net_flow']         = df_clean['pickup_count'] - df_clean['dropoff_count']
traffic_num = df_clean['traffic_level'].map({'low': 1, 'medium': 2, 'high': 3})
df_clean['demand_traffic_ratio'] = df_clean['pickup_count'] / traffic_num

# Target variable
thresholds = df_clean['demand_intensity'].quantile([0.33, 0.66])
df_clean['demand_label'] = pd.cut(
    df_clean['demand_intensity'],
    bins=[-np.inf, thresholds[0.33], thresholds[0.66], np.inf],
    labels=['LOW', 'MEDIUM', 'HIGH']
)

# Encode categoricals
le_traffic = LabelEncoder()
le_zone    = LabelEncoder()
df_clean['traffic_encoded'] = le_traffic.fit_transform(df_clean['traffic_level'])
df_clean['zone_encoded']    = le_zone.fit_transform(df_clean['zone_id'])

df_clean.dropna(subset=['demand_intensity', 'demand_label'], inplace=True)
df_clean.reset_index(drop=True, inplace=True)
df_clean.to_csv(PROCESSED_CSV, index=False)
print(f"  ✓ Missing values after cleaning: {df_clean.isnull().sum().sum()}")
print(f"  ✓ New features added: peak_hour_flag, weekend_flag,")
print(f"    demand_intensity, net_flow, demand_traffic_ratio")
print(f"  ✓ ML features exclude current-demand-derived leakage fields")
print(f"  ✓ Target distribution:\n{df_clean['demand_label'].value_counts().to_string()}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 4: EXPLORATORY DATA ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[STEP 4] Running EDA & Generating Charts...")

fig = plt.figure(figsize=(20, 22))
fig.patch.set_facecolor('#0f1117')
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
hourly = df_clean.groupby('hour_of_day')['pickup_count'].mean()
colors = [ACCENT4 if (7 <= h <= 9) or (17 <= h <= 19) else ACCENT for h in hourly.index]
ax1.bar(hourly.index, hourly.values, color=colors, edgecolor='none', width=0.8, alpha=0.9)
ax1.set_title('Avg Pickups by Hour of Day', fontsize=14, fontweight='bold', color='white', pad=12)
ax1.set_xlabel('Hour of Day', fontsize=11)
ax1.set_ylabel('Avg Pickup Count', fontsize=11)
ax1.axvspan(7, 9, alpha=0.08, color=ACCENT4, label='Peak Hours')
ax1.axvspan(17, 19, alpha=0.08, color=ACCENT4)
ax1.legend(fontsize=9, framealpha=0.2)
ax1.grid(axis='y', alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
zone_demand = (df_clean.groupby('zone_id')['demand_intensity']
               .mean().sort_values(ascending=False).head(10))
bars2 = ax2.barh(zone_demand.index, zone_demand.values,
                  color=PALETTE[:len(zone_demand)], edgecolor='none', alpha=0.9)
ax2.set_title('Top 10 Zones by Avg Demand Intensity', fontsize=14, fontweight='bold', color='white', pad=12)
ax2.set_xlabel('Avg Demand Intensity', fontsize=11)
ax2.invert_yaxis()
for bar, val in zip(bars2, zone_demand.values):
    ax2.text(val + 0.3, bar.get_y() + bar.get_height()/2,
             f'{val:.0f}', va='center', fontsize=9, color='white')
ax2.grid(axis='x', alpha=0.3)

ax3 = fig.add_subplot(gs[1, 0])
peak_data = df_clean.groupby('peak_hour_flag')['pickup_count'].mean()
labels = ['Non-Peak', 'Peak Hour']
wedges, texts, autotexts = ax3.pie(
    peak_data.values, labels=labels,
    colors=[ACCENT, ACCENT4], autopct='%1.1f%%',
    startangle=90, pctdistance=0.75,
    wedgeprops=dict(width=0.55, edgecolor='#0f1117', linewidth=3)
)
for t in texts:     t.set_color('white'); t.set_fontsize(11)
for t in autotexts: t.set_color('white'); t.set_fontsize(11); t.set_fontweight('bold')
ax3.set_title('Peak vs Non-Peak Demand Share', fontsize=14, fontweight='bold', color='white', pad=12)

ax4 = fig.add_subplot(gs[1, 1])
days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
daily = df_clean.groupby('day_of_week')['pickup_count'].mean()
day_colors = [ACCENT4 if d >= 5 else ACCENT for d in daily.index]
ax4.bar(days, daily.values, color=day_colors, edgecolor='none', width=0.7, alpha=0.9)
ax4.set_title('Avg Pickups by Day of Week', fontsize=14, fontweight='bold', color='white', pad=12)
ax4.set_xlabel('Day', fontsize=11)
ax4.set_ylabel('Avg Pickup Count', fontsize=11)
ax4.grid(axis='y', alpha=0.3)

ax5 = fig.add_subplot(gs[2, 0])
traffic_demand = df_clean.groupby('traffic_level')['pickup_count'].mean()
ax5.bar(traffic_demand.index, traffic_demand.values,
        color=[ACCENT2, ACCENT3, ACCENT4], edgecolor='none', width=0.5, alpha=0.9)
ax5.set_title('Avg Pickups by Traffic Level', fontsize=14, fontweight='bold', color='white', pad=12)
ax5.set_xlabel('Traffic Level', fontsize=11)
ax5.set_ylabel('Avg Pickup Count', fontsize=11)
ax5.grid(axis='y', alpha=0.3)

ax6 = fig.add_subplot(gs[2, 1])
corr_cols = ['pickup_count','dropoff_count','temperature','rain',
             'peak_hour_flag','weekend_flag','demand_intensity',
             'traffic_encoded','event_flag']
corr = df_clean[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, ax=ax6, mask=mask, cmap='RdYlGn',
            annot=True, fmt='.2f', annot_kws={'size': 7},
            linewidths=0.5, linecolor='#0f1117', cbar_kws={'shrink': 0.8})
ax6.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', color='white', pad=12)
ax6.tick_params(axis='x', rotation=45, labelsize=8)
ax6.tick_params(axis='y', rotation=0,  labelsize=8)

fig.suptitle('Urban Mobility — Exploratory Data Analysis  |  2024',
             fontsize=18, fontweight='bold', color='white', y=0.98)

plt.savefig(EDA_PNG, bbox_inches='tight', facecolor='#0f1117', dpi=140)
plt.close()
print(f"  ✓ EDA charts saved → notebooks/eda_charts.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 5: ML MODEL — RANDOM FOREST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[STEP 5] Training Random Forest Classifier...")

FEATURES = [
    'hour_of_day', 'day_of_week', 'month', 'quarter',
    'temperature', 'rain', 'event_flag',
    'peak_hour_flag', 'weekend_flag',
    'zone_encoded', 'traffic_encoded',
]

X = df_clean[FEATURES]
y = df_clean['demand_label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

param_grid = {
    'n_estimators':     [100, 200],
    'max_depth':        [10, 15, None],
    'min_samples_split':[2, 5],
}

rf_base = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
grid_search = GridSearchCV(rf_base, param_grid, cv=3,
                            scoring='f1_weighted', n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print(f"  ✓ Best params: {grid_search.best_params_}")

y_pred = best_rf.predict(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec  = recall_score(y_test, y_pred, average='weighted')
f1   = f1_score(y_test, y_pred, average='weighted')

print(f"\n  ┌─ MODEL EVALUATION ──────────────────────┐")
print(f"  │  Accuracy  : {acc:.4f}  ({acc*100:.1f}%)          │")
print(f"  │  Precision : {prec:.4f}                      │")
print(f"  │  Recall    : {rec:.4f}                      │")
print(f"  │  F1-Score  : {f1:.4f}                      │")
print(f"  └─────────────────────────────────────────┘")

fig2, axes = plt.subplots(1, 2, figsize=(16, 6))
fig2.patch.set_facecolor('#0f1117')

cm = confusion_matrix(y_test, y_pred, labels=['LOW','MEDIUM','HIGH'])
labels = ['LOW', 'MEDIUM', 'HIGH']
im = axes[0].imshow(cm, cmap='plasma', interpolation='nearest')
axes[0].set_xticks(range(3)); axes[0].set_yticks(range(3))
axes[0].set_xticklabels(labels, fontsize=12)
axes[0].set_yticklabels(labels, fontsize=12)
axes[0].set_xlabel('Predicted Label', fontsize=12)
axes[0].set_ylabel('True Label', fontsize=12)
axes[0].set_title('Confusion Matrix — Random Forest', fontsize=14,
                   fontweight='bold', color='white', pad=12)
for i in range(3):
    for j in range(3):
        axes[0].text(j, i, str(cm[i, j]), ha='center', va='center',
                     fontsize=16, fontweight='bold',
                     color='white' if cm[i,j] < cm.max()*0.6 else '#0f1117')
plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

importances = best_rf.feature_importances_
feat_df = (pd.DataFrame({'Feature': FEATURES, 'Importance': importances})
             .sort_values('Importance', ascending=True).tail(12))
colors_fi = plt.cm.plasma(np.linspace(0.3, 0.9, len(feat_df)))
axes[1].barh(feat_df['Feature'], feat_df['Importance'],
              color=colors_fi, edgecolor='none', alpha=0.9)
axes[1].set_title('Feature Importance — Top Features', fontsize=14,
                   fontweight='bold', color='white', pad=12)
axes[1].set_xlabel('Importance Score', fontsize=12)
axes[1].grid(axis='x', alpha=0.3)

fig2.suptitle('Random Forest — Model Evaluation & Feature Importance',
              fontsize=15, fontweight='bold', color='white', y=1.01)
plt.tight_layout()
plt.savefig(MODEL_PNG, bbox_inches='tight', facecolor='#0f1117', dpi=140)
plt.close()
print(f"  ✓ Model charts saved → notebooks/model_evaluation.png")

joblib.dump(best_rf, MODEL_FILE)
print(f"  ✓ Model saved → models/random_forest.joblib")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 6: DEMAND SCENARIO FORECASTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[STEP 6] Generating 24-Hour Demand Scenario Forecast...")

def build_forecast_record(zone_id, hour, dow, month, temp=22.0,
                           rain=0, event=0, traffic='medium'):
    return {
        'hour_of_day'         : hour,
        'day_of_week'         : dow,
        'month'               : month,
        'quarter'             : (month - 1) // 3 + 1,
        'temperature'         : temp,
        'rain'                : rain,
        'event_flag'          : event,
        'peak_hour_flag'      : 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0,
        'weekend_flag'        : 1 if dow >= 5 else 0,
        'zone_encoded'        : le_zone.transform([zone_id])[0],
        'traffic_encoded'     : le_traffic.transform([traffic])[0],
    }

forecast_records = []
np.random.seed(7)
for zone in ZONES:
    zone_event_hours = set(np.random.choice(range(24), size=3, replace=False))
    for hour in range(24):
        if (7 <= hour <= 9) or (17 <= hour <= 19):
            traffic = np.random.choice(['medium', 'high'], p=[0.3, 0.7])
        elif 0 <= hour <= 5:
            traffic = 'low'
        else:
            traffic = np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2])
        event = 1 if hour in zone_event_hours else 0
        temp  = 22 + np.random.normal(0, 5)
        rain  = np.random.choice([0, 1], p=[0.8, 0.2])
        rec = build_forecast_record(zone, hour, dow=0, month=5,
                                     temp=temp, rain=rain,
                                     event=event, traffic=traffic)
        rec['zone_id'] = zone
        forecast_records.append(rec)

forecast_df = pd.DataFrame(forecast_records)
forecast_input = forecast_df[FEATURES]
forecast_df['predicted_demand'] = best_rf.predict(forecast_input)
forecast_df['demand_proba']     = best_rf.predict_proba(forecast_input).max(axis=1)

high_demand = (forecast_df[forecast_df['predicted_demand'] == 'HIGH']
               .groupby('zone_id')
               .agg(high_demand_hours=('predicted_demand','count'),
                    avg_confidence=('demand_proba','mean'))
               .sort_values('high_demand_hours', ascending=False)
               .reset_index())

print("  ✓ Top 5 High-Demand Zones (24h Scenario Forecast):")
print(f"  {'Rank':<5} {'Zone':<12} {'High-Demand Hours':<20} {'Avg Confidence'}")
print(f"  {'-'*5} {'-'*12} {'-'*20} {'-'*14}")
for rank, row in high_demand.head(5).iterrows():
    print(f"  {rank+1:<5} {row['zone_id']:<12} "
          f"{row['high_demand_hours']:<20} {row['avg_confidence']:.2%}")

forecast_df.to_csv(FORECAST_CSV, index=False)
high_demand.to_csv(TOP_ZONES_CSV, index=False)
print(f"  ✓ Forecast saved → data/forecast_output.csv")

fig3, axes3 = plt.subplots(1, 2, figsize=(18, 6))
fig3.patch.set_facecolor('#0f1117')

demand_map = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}

sample_zones = ZONES[:5]
for zone in sample_zones:
    z_data = forecast_df[forecast_df['zone_id'] == zone].copy()
    z_data['demand_num'] = z_data['predicted_demand'].map(demand_map)
    axes3[0].plot(z_data['hour_of_day'], z_data['demand_num'],
                  marker='o', markersize=4, alpha=0.8, label=zone)

axes3[0].set_title('24-Hour Demand Scenario Forecast (Sample Zones)', fontsize=14,
                    fontweight='bold', color='white', pad=12)
axes3[0].set_xlabel('Hour of Day', fontsize=11)
axes3[0].set_ylabel('Demand Level (1=LOW 2=MED 3=HIGH)', fontsize=11)
axes3[0].set_yticks([1, 2, 3]); axes3[0].set_yticklabels(['LOW','MEDIUM','HIGH'])
axes3[0].legend(fontsize=9, framealpha=0.2)
axes3[0].grid(alpha=0.3)

if len(high_demand) > 0:
    top10_zones = high_demand.head(10)
    colors_bar  = plt.cm.plasma(np.linspace(0.3, 0.9, len(top10_zones)))
    axes3[1].barh(top10_zones['zone_id'], top10_zones['high_demand_hours'],
                   color=colors_bar, edgecolor='none', alpha=0.9)
    axes3[1].set_title('Top Zones by High-Demand Hours (24h Scenario)', fontsize=14,
                        fontweight='bold', color='white', pad=12)
    axes3[1].set_xlabel('Number of High-Demand Hours', fontsize=11)
    axes3[1].invert_yaxis()
    axes3[1].grid(axis='x', alpha=0.3)

fig3.suptitle('Demand Forecasting — 24-Hour Scenario Across All Zones',
              fontsize=15, fontweight='bold', color='white', y=1.01)
plt.tight_layout()
plt.savefig(FORECAST_PNG, bbox_inches='tight', facecolor='#0f1117', dpi=140)
plt.close()
print(f"  ✓ Forecast charts saved → notebooks/forecast_charts.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FINAL SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 65)
print("  PIPELINE COMPLETE — SUMMARY")
print("=" * 65)
print(f"  Dataset         : {df_clean.shape[0]:,} records x {df_clean.shape[1]} columns")
print(f"  Training size   : {len(X_train):,} records")
print(f"  Test size       : {len(X_test):,} records")
print(f"  Model Accuracy  : {acc*100:.1f}%")
print(f"  F1 Score        : {f1:.4f}")
print(f"  Zones Scenario  : {df_clean['zone_id'].nunique()} zones, 24 hours")
print("=" * 65)
print(f"  All files saved to: {SCRIPT_DIR}")
print(f"  data/raw_mobility_data.csv")
print(f"  data/processed_data.csv")
print(f"  data/forecast_output.csv")
print(f"  data/top_zones_forecast.csv")
print(f"  notebooks/eda_charts.png")
print(f"  notebooks/model_evaluation.png")
print(f"  notebooks/forecast_charts.png")
print(f"  models/random_forest.joblib")
print("=" * 65)
