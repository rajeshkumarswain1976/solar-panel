

# Step 1: Imports
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ===============================
# Step 2: Config
# ===============================
INPUT_PATH = "final_solar_dataset_cleaned_large.csv"
OUTPUT_DIR = "solar_analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COL = "AC_POWER"
UNDERPERF_THRESHOLD = 0.20
RANDOM_STATE = 42

# ===============================
# Step 3: Load Data
# ===============================
df = pd.read_csv(INPUT_PATH)
print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")

# Parse datetime if exists
for c in df.columns:
    if "date" in c.lower() or "time" in c.lower():
        try:
            df[c] = pd.to_datetime(df[c])
            datetime_col = c
            break
        except:
            datetime_col = None
else:
    datetime_col = None

# ===============================
# Step 4: Abnormal Value Detection
# ===============================
def detect_outliers_iqr(series, k=1.5):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - k*iqr, q3 + k*iqr
    return (series < low) | (series > high)

def detect_outliers_isoforest(df_num):
    iso = IsolationForest(contamination=0.02, random_state=RANDOM_STATE)
    preds = iso.fit_predict(df_num.fillna(0))
    return preds == -1

num_cols = df.select_dtypes(include=[np.number]).columns
outlier_mask = pd.DataFrame(False, index=df.index, columns=num_cols)
for col in num_cols:
    outlier_mask[col] |= detect_outliers_iqr(df[col])
iso_mask = detect_outliers_isoforest(df[num_cols])
outlier_mask["IForest"] = iso_mask

# Combine masks
df["any_outlier"] = outlier_mask.any(axis=1)
print(f"Abnormal rows detected: {df['any_outlier'].sum()}")

# Handle abnormal values (cap them)
for col in num_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df[col] = np.clip(df[col], low, high)

# ===============================
# Step 5: Feature Engineering
# ===============================
if datetime_col:
    df["hour"] = df[datetime_col].dt.hour
    df["month"] = df[datetime_col].dt.month
    df["day"] = df[datetime_col].dt.day
    df["dayofyear"] = df[datetime_col].dt.dayofyear

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

# ===============================
# Step 6: Model Training
# ===============================
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", num_pipe, num_features),
    ("cat", cat_pipe, cat_features)
])

model = Pipeline([
    ("prep", preprocess),
    ("rf", RandomForestRegressor(
        n_estimators=120,
        random_state=RANDOM_STATE,
        n_jobs=-1))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=True)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# ===============================
# Step 7: Evaluation
# ===============================
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f" Model Performance:")
print(f"  MAE  = {mae:.5f}")
print(f"  RMSE = {rmse:.5f}")
print(f"  R²   = {r2:.5f}")

# ===============================
# Step 8: Detect Performance Dips
# ===============================
results = X_test.copy()
results["actual"] = y_test
results["predicted"] = y_pred
results["residual"] = results["actual"] - results["predicted"]
results["underperf_pct"] = (results["predicted"] - results["actual"]) / results["predicted"]
results["underperf_flag"] = results["underperf_pct"] > UNDERPERF_THRESHOLD

print(f"Underperformance rows: {results['underperf_flag'].sum()}")

# ===============================
# Step 9: Rule-Based Cause Detection
# ===============================
def detect_cause(row):
    causes = []
    if row.get("SHADING_FACTOR", 0) > 0.5:
        causes.append("High shading")
    if row.get("DUST_ACCUMULATION", 0) > 0.4:
        causes.append("Panel soiling / dust buildup")
    if row.get("CLEANING_ALERT", 0) == 1:
        causes.append("Missed cleaning schedule")
    if row.get("EFFICIENCY_DROP", 0) > 0.2:
        causes.append("Efficiency degradation")
    if row.get("TILT_ADJUSTMENT_ALERT", 0) == 1:
        causes.append("Tilt misalignment")
    if row.get("PERFORMANCE_ALERT", 0) == 1:
        causes.append("System / inverter issue")
    if row.get("PRECIPITATION", 0) > 10:
        causes.append("Weather impact (rain/cloud)")
    return "; ".join(causes) if causes else "No clear cause"

results["likely_cause"] = results.apply(detect_cause, axis=1)

# ===============================
# Step 10: Recommendations
# ===============================
def recommend_action(cause_str):
    if "shading" in cause_str.lower():
        return "Inspect site for shading; trim or remove obstacles."
    if "dust" in cause_str.lower():
        return "Schedule immediate panel cleaning."
    if "cleaning" in cause_str.lower():
        return "Verify cleaning schedule and trigger cleaning."
    if "efficiency" in cause_str.lower():
        return "Check inverter and panel output for degradation."
    if "tilt" in cause_str.lower():
        return "Recalibrate tracker or adjust tilt."
    if "system" in cause_str.lower():
        return "Inspect inverter/system components for faults."
    if "weather" in cause_str.lower():
        return "No action needed; natural variability."
    return "Monitor performance for persistent dips."

results["recommendation"] = results["likely_cause"].apply(recommend_action)

# ===============================
# Step 10b: Environmental & Observational Factor Detection
# ===============================
def detect_environmental_factors(row):
    factors = []
    # Low irradiance / cloudy
    irr_cols = [c for c in ["IRRADIANCE", "GHI", "POA_IRRADIANCE", "SOLAR_RADIATION", "GLOBAL_HORIZONTAL_IRRADIANCE"] if c in row.index]
    if irr_cols:
        irr_vals = [row.get(c, np.nan) for c in irr_cols]
        try:
            irr_min = np.nanmin(irr_vals)
            if not np.isnan(irr_min) and irr_min < 100:  # W/m2 threshold for low sun
                factors.append("Low irradiance / cloudy")
        except:
            pass

    # High ambient / module temperature
    if "MODULE_TEMPERATURE" in row.index and pd.notnull(row.get("MODULE_TEMPERATURE")):
        if row.get("MODULE_TEMPERATURE") > 45:
            factors.append("High module temperature")
    if "AMBIENT_TEMPERATURE" in row.index and pd.notnull(row.get("AMBIENT_TEMPERATURE")):
        if row.get("AMBIENT_TEMPERATURE") > 40:
            factors.append("High ambient temperature")

    # Precipitation / rain
    if "PRECIPITATION" in row.index and pd.notnull(row.get("PRECIPITATION")):
        if row.get("PRECIPITATION") > 0.1:
            factors.append("Precipitation / rain")

    # Cloud cover
    if "CLOUD_COVER" in row.index and pd.notnull(row.get("CLOUD_COVER")):
        if row.get("CLOUD_COVER") > 0.6:
            factors.append("High cloud cover")

    # High wind
    if "WIND_SPEED" in row.index and pd.notnull(row.get("WIND_SPEED")):
        if row.get("WIND_SPEED") > 15:
            factors.append("High wind")

    return "; ".join(factors) if factors else "None detected"


def detect_observational_factors(row):
    obs = []
    # Shading / soiling / cleaning
    if row.get("SHADING_FACTOR", 0) > 0.5:
        obs.append("Shading")
    if row.get("DUST_ACCUMULATION", 0) > 0.4:
        obs.append("Soiling / dust")
    if row.get("CLEANING_ALERT", 0) == 1:
        obs.append("Missed cleaning")

    # Tilt / alignment / tracker
    if row.get("TILT_ADJUSTMENT_ALERT", 0) == 1:
        obs.append("Tilt misalignment")

    # Inverter/system alerts
    if row.get("PERFORMANCE_ALERT", 0) == 1:
        obs.append("Inverter/system alert")

    # Sensor or missing data issues
    if "SENSOR_STATUS" in row.index and row.get("SENSOR_STATUS") in ["fault", "error", 0]:
        obs.append("Sensor fault")
    if row.get("any_outlier", False):
        obs.append("Anomalous sensor reading / outlier")

    # Residual-based observational flag
    if "underperf_flag" in row.index and row.get("underperf_flag"):
        obs.append("Observed underperformance")

    return "; ".join(obs) if obs else "None detected"


def recommend_actions_from_factors(env_str, obs_str):
    actions = []
    s = (env_str + "; " + obs_str).lower()
    if "low irradiance" in s or "cloud" in s:
        actions.append("No immediate site action; consider forecasting or re-scheduling maintenance during low-irradiance periods.")
    if "high module temperature" in s or "high ambient" in s:
        actions.append("Inspect ventilation/cooling, verify derating curves and check for hotspoting.")
    if "precipitation" in s:
        actions.append("Allow system to self-recover after rain; inspect for wet soiling or runoff issues if persistent.")
    if "high wind" in s:
        actions.append("Inspect mounting and structural integrity; secure loose components.")
    if "shading" in s:
        actions.append("Inspect and remove nearby shading (trees/buildings) or consider module-level mitigation (optimizers).")
    if "soiling" in s or "dust" in s:
        actions.append("Schedule panel cleaning and review cleaning frequency.")
    if "missed cleaning" in s:
        actions.append("Trigger cleaning workflow and verify completion.")
    if "tilt" in s or "misalignment" in s:
        actions.append("Run alignment check and recalibrate tracker/tilt settings.")
    if "inverter" in s or "system alert" in s:
        actions.append("Open maintenance ticket for inverter/system diagnostics.")
    if "sensor fault" in s or "anomalous" in s:
        actions.append("Validate sensor health, replace or recalibrate sensors as needed.")
    if "observed underperformance" in s and not actions:
        actions.append("Investigate underperformance: start with visual inspection and logs review.")

    # fallback
    if not actions:
        actions.append("Monitor and collect more data; escalate if persistent.")

    # dedupe while preserving order
    seen = set()
    dedup = []
    for a in actions:
        if a not in seen:
            dedup.append(a)
            seen.add(a)
    return "; ".join(dedup)


# Apply detectors and produce a compact output CSV with recommendations
results["environmental_factors"] = results.apply(detect_environmental_factors, axis=1)
results["observational_factors"] = results.apply(detect_observational_factors, axis=1)
results["recommended_actions_combined"] = results.apply(lambda r: recommend_actions_from_factors(r["environmental_factors"], r["observational_factors"]), axis=1)

# Save an additional focused CSV
focus_cols = [c for c in [datetime_col if datetime_col else None, "actual", "predicted", "underperf_pct", "underperf_flag", "environmental_factors", "observational_factors", "recommended_actions_combined"] if c]
results[focus_cols].to_csv(f"{OUTPUT_DIR}/environmental_observational_factors.csv", index=False)
print(f"💾 Environmental & observational factors saved to: {OUTPUT_DIR}/environmental_observational_factors.csv")

# ===============================
# Step 11: Save & Visualize
# ===============================
results.to_csv(f"{OUTPUT_DIR}/solar_predictions_diagnostics.csv", index=False)
print(f"💾 Results saved to: {OUTPUT_DIR}/solar_predictions_diagnostics.csv")

# Plot predicted vs actual
plt.figure(figsize=(7,7))
sns.scatterplot(x="predicted", y="actual", data=results, alpha=0.5)
plt.plot([results["actual"].min(), results["actual"].max()],
         [results["actual"].min(), results["actual"].max()],
         'r--')
plt.title("Predicted vs Actual AC Power")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Show top underperforming cases
top_dips = results[results["underperf_flag"]].sort_values("underperf_pct", ascending=False).head(10)
display(top_dips[["actual","predicted","underperf_pct","likely_cause","recommendation"]])

# ===============================
# ✅ END OF NOTEBOOK
# ===============================
