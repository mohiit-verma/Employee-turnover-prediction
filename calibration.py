import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# Assuming your data looks like this:
df_train = pd.DataFrame({
    'employee_id': [1, 2, 3, 4, 5],
    'duration': [45, 120, 400, 365, 200],  # Days from hire to event/censoring
    'event': [1, 1, 0, 1, 1],              # 1=left, 0=still employed
    'feature1': [0.5, 0.8, 0.2, 0.9, 0.3],
    'feature2': [1.2, 2.1, 0.8, 1.8, 1.0]
})

# Step 1: Train your AFT model (as you normally would)
X_train = df_train[['feature1', 'feature2']].values
y_train_duration = df_train['duration'].values
y_train_event = df_train['event'].values

# Train AFT model
aft_model = xgb.XGBRegressor(
    objective='survival:aft',
    eval_metric='aft-nloglik'
)
# Note: XGBoost survival requires specific data format - adjust as needed

# Step 2: Create binary labels for calibration
time_horizon = 365  # 12 months
y_train_binary, valid_indices = create_binary_labels(
    y_train_duration, y_train_event, time_horizon
)

# Filter to valid samples only
X_train_valid = X_train[valid_indices]
y_binary_valid = y_train_binary

# Step 3: Get uncalibrated probabilities from AFT model
def aft_to_probability(aft_model, X, time_horizon):
    """Convert AFT predictions to survival probabilities"""
    # Get predicted survival times
    pred_survival_times = aft_model.predict(X)
    
    # Convert to probability of event before time_horizon
    # Using exponential survival: P(T <= t) = 1 - exp(-t/λ)
    probabilities = 1 - np.exp(-time_horizon / pred_survival_times)
    
    # Clip to valid probability range
    return np.clip(probabilities, 0.001, 0.999)

uncalibrated_probs = aft_to_probability(aft_model, X_train_valid, time_horizon)

# Step 4: Fit Platt calibrator
platt_calibrator = LogisticRegression()
platt_calibrator.fit(uncalibrated_probs.reshape(-1, 1), y_binary_valid)

# Step 5: Get calibrated probabilities
calibrated_probs = platt_calibrator.predict_proba(
    uncalibrated_probs.reshape(-1, 1)
)[:, 1]

print("Comparison:")
print("Employee | Uncalibrated | Calibrated | Actual")
print("-" * 45)
for i in range(len(calibrated_probs)):
    print(f"{valid_indices[i]:8d} | {uncalibrated_probs[i]:11.3f} | {calibrated_probs[i]:10.3f} | {y_binary_valid[i]:6d}")


# Create binary labels for multiple horizons
horizons = [90, 180, 365, 730]  # 3, 6, 12, 24 months

binary_labels = {}
for horizon in horizons:
    binary_labels[horizon], valid_idx = create_binary_labels(
        durations, events, horizon
    )
    
# Train separate calibrators for each horizon
calibrators = {}
for horizon in horizons:
    y_binary = binary_labels[horizon]
    uncal_probs = aft_to_probability(aft_model, X_train[valid_idx], horizon)
    
    calibrators[horizon] = LogisticRegression()
    calibrators[horizon].fit(uncal_probs.reshape(-1, 1), y_binary)