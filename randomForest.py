import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from FrameDataset import SquatKneeFrameDataset
import numpy as np
import joblib

def add_temporal_features(data, column_index=0, lag=3):
    """
    Add temporal features such as lagged features and rolling statistics
    to a NumPy array.
    
    Args:
        data (np.ndarray): The data matrix (2D array with features).
        column_index (int): The index of the column for which to add temporal features (e.g., knee_angle).
        lag (int): The number of lag features and the window for rolling statistics.
        
    Returns:
        np.ndarray: The data with added temporal features.
    """
    # Copy original data to avoid modifying in place
    data = data.copy()

    # Lag features (shift the data by i positions)
    for i in range(1, lag + 1):
        lagged_column = np.roll(data[:, column_index], i)  # Shift the column values by i
        data = np.column_stack((data, lagged_column))

    # Rolling statistics (mean, std) with valid mode
    rolling_mean = np.convolve(data[:, column_index], np.ones(lag) / lag, mode='valid')
    rolling_mean = np.pad(rolling_mean, (lag-1, 0), mode='constant', constant_values=np.nan)

    # Padding to handle window size mismatch
    # rolling_std = np.array([np.std(data[max(0, i - lag + 1):i + 1, column_index]) for i in range(len(data))])
    # rolling_std = np.pad(rolling_std, (lag-1, 0), mode='constant', constant_values=np.nan)

    # Temporal difference (angle_diff)
    # angle_diff = np.diff(data[:, column_index], prepend=data[0, column_index])
    
    # # Pad temporal difference with NaN to align sizes
    # angle_diff = np.pad(angle_diff, (0, 1), mode='constant', constant_values=np.nan)

    # Now make sure all arrays have the same length
    assert len(data) == len(rolling_mean) #== len(rolling_std) == len(angle_diff), "Arrays have mismatched lengths"

    # Concatenate all features to the data (stack columns)
    data = np.column_stack((data, rolling_mean)) #rolling_std, angle_diff))
    
    return data


# Load your dataset (already pre-processed)
train_dataset = SquatKneeFrameDataset("Squat_Train.csv", threshold_pct=50, sigma=2.0)
test_dataset = SquatKneeFrameDataset("Squat_Test.csv", threshold_pct=50, sigma=2.0)
# Extract the features and labels
X_train = train_dataset.data  # Feature matrix (angles)
y_train = train_dataset.labels  # Labels (UP or DOWN)
X_test = test_dataset.data  # Feature matrix (angles)
y_test = test_dataset.labels  # Labels (UP or DOWN)

# --- RANDOM FOREST HYPERPARAMETER TUNING ---
# Set up parameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']  # Fixed this part
}

# Initialize the Random Forest model
rf = RandomForestClassifier(random_state=42)

# Set up TimeSeriesSplit or GroupKFold (if needed)
tscv = TimeSeriesSplit(n_splits=5)  # Temporal cross-validation

# Perform grid search to find the best hyperparameters for Random Forest
rf_grid_search = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=tscv, scoring='accuracy', n_jobs=-1, verbose=1)
rf_grid_search.fit(X_train, y_train)

# Best model from GridSearchCV
best_rf_model = rf_grid_search.best_estimator_

# Step 2: Evaluate Random Forest model
rf_y_pred = best_rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print(f"\nRandom Forest Model Accuracy: {rf_accuracy * 100:.2f}%")
print(f"Best Random Forest Hyperparameters: {rf_grid_search.best_params_}")

# --- XGBoost HYPERPARAMETER TUNING ---
# Set up parameter grid for XGBoost
xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1, 0.2]
}

# Initialize the XGBoost model
xgb = XGBClassifier(random_state=42)

# Perform grid search to find the best hyperparameters for XGBoost
xgb_grid_search = GridSearchCV(estimator=xgb, param_grid=xgb_param_grid, cv=tscv, scoring='accuracy', n_jobs=-1, verbose=1)
xgb_grid_search.fit(X_train, y_train)

# Best model from GridSearchCV
best_xgb_model = xgb_grid_search.best_estimator_

# Step 3: Evaluate XGBoost model
xgb_y_pred = best_xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_y_pred)
print(f"\nXGBoost Model Accuracy: {xgb_accuracy * 100:.2f}%")
print(f"Best XGBoost Hyperparameters: {xgb_grid_search.best_params_}")

# --- EVALUATION METRICS ---
# Random Forest Metrics
print("\nRandom Forest Model Evaluation:")
print(f"Precision: {precision_score(y_test, rf_y_pred)}")
print(f"Recall: {recall_score(y_test, rf_y_pred)}")
print(f"F1-Score: {f1_score(y_test, rf_y_pred)}")

# XGBoost Metrics
print("\nXGBoost Model Evaluation:")
print(f"Precision: {precision_score(y_test, xgb_y_pred)}")
print(f"Recall: {recall_score(y_test, xgb_y_pred)}")
print(f"F1-Score: {f1_score(y_test, xgb_y_pred)}")

# --- CONFUSION MATRIX ---
# Random Forest Confusion Matrix
rf_cm = confusion_matrix(y_test, rf_y_pred)
plt.figure(figsize=(10, 5))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', xticklabels=["DOWN", "UP"], yticklabels=["DOWN", "UP"])
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# XGBoost Confusion Matrix
xgb_cm = confusion_matrix(y_test, xgb_y_pred)
plt.figure(figsize=(10, 5))
sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Blues', xticklabels=["DOWN", "UP"], yticklabels=["DOWN", "UP"])
plt.title('XGBoost Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Get feature importance from Random Forest
rf_feature_importance = best_rf_model.feature_importances_

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(range(len(rf_feature_importance)), rf_feature_importance, align='center')
plt.yticks(range(len(rf_feature_importance)), [f"Feature {i}" for i in range(X_train.shape[1])])
plt.xlabel('Importance')
plt.title('Random Forest Feature Importance')
plt.show()

# Plot feature importance for XGBoost
plot_importance(best_xgb_model, importance_type='weight', max_num_features=10, height=0.6)
plt.title('XGBoost Feature Importance')
plt.show()

# --- PLOT RANDOM FOREST VALIDATION ACCURACY OVER GRID SEARCH ---

# Extract mean validation scores for each hyperparameter combination
rf_results = rf_grid_search.cv_results_
mean_test_scores = rf_results['mean_test_score']
params = rf_results['params']

# Create a DataFrame for easier plotting
rf_scores_df = pd.DataFrame(params)
rf_scores_df['mean_test_score'] = mean_test_scores

# Sort by validation score
rf_scores_df = rf_scores_df.sort_values(by='mean_test_score', ascending=False)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(range(len(rf_scores_df)), rf_scores_df['mean_test_score'], marker='o')
plt.title('Random Forest Validation Accuracy over Hyperparameter Search')
plt.xlabel('Hyperparameter Combination')
plt.ylabel('Validation Accuracy')
plt.grid()
plt.show()

# Save the best Random Forest model
joblib.dump(best_rf_model, "models/best_rf_model.pkl")
print("Random Forest model saved successfully!")

# Extract mean validation scores for each hyperparameter combination
xgb_results = xgb_grid_search.cv_results_
mean_test_scores = xgb_results['mean_test_score']
params = xgb_results['params']

# Create a DataFrame for easier plotting
xgb_scores_df = pd.DataFrame(params)
xgb_scores_df['mean_test_score'] = mean_test_scores

# Sort by validation score
xgb_scores_df = xgb_scores_df.sort_values(by='mean_test_score', ascending=False)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(range(len(xgb_scores_df)), xgb_scores_df['mean_test_score'], marker='o')
plt.title('Random Forest Validation Accuracy over Hyperparameter Search')
plt.xlabel('Hyperparameter Combination')
plt.ylabel('Validation Accuracy')
plt.grid()
plt.show()

# Save the best Random Forest model
joblib.dump(best_xgb_model, "models/best_xgb_model.pkl")
print("Random Forest model saved successfully!")

# Save it
joblib.dump(train_dataset.scaler, "models/scaler.pkl")
print("Scaler saved successfully!")

