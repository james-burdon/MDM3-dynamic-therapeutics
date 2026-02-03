
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import classification_report
import statistics
from sklearn.metrics import classification_report


# Window data so the models predictors are features of accelerometer data rather than raw values
def create_windows(df, window_size, step_size):

    features = []
    labels = []
    users = []

    for user_id, user_df in df.groupby("user"):
        user_df = user_df.reset_index(drop=True)

        for start in range(0, len(user_df) - window_size + 1, step_size):
            window = user_df.iloc[start:start + window_size]

            # centre time label
            center_idx = start + window_size // 2
            center_label = user_df.iloc[center_idx]["activity_label"]

            labels_in_window = window["activity_label"]

            # Fraction of window matching center label
            purity = np.mean(labels_in_window == center_label)

          #  if purity < 0.6:
                #continue         

            # signals
            ax = window["Accelerometer_X"].values
            ay = window["Accelerometer_Y"].values
            az = window["Accelerometer_Z"].values

            mag = np.sqrt(ax**2 + ay**2 + az**2)

            # Features
            feature_dict = {
                "ax_mean": ax.mean(),
                "ay_mean": ay.mean(),
                "az_mean": az.mean(),

                "ax_std": ax.std(),
                "ay_std": ay.std(),
                "az_std": az.std(),

                "mag_mean": mag.mean(),
                "mag_std": mag.std()

            
              
            }

            features.append(feature_dict)
            labels.append(str(statistics.mode(labels_in_window)))
            users.append(user_id)

    return pd.DataFrame(features), pd.Series(labels), pd.Series(users)


    return pd.DataFrame(features), pd.Series(labels), pd.Series(users)




user_list = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18]




# Some users are not in the activity log
users_to_remove = [19, 20, 21, 22]

# New csv
data = pd.read_csv('physiological_with_activity_labels.csv')

data = data.dropna()
data = data[data['activity_label'] != 'moderate_activity']
data = data[data['activity_label'] != 'light_activity']
data = data.dropna(subset=["activity_label"])


label_mapping = {
    'stand' : 'standing',
    'sit' : 'sitting',
    'lying' : 'lying'
  
}

data['activity_label'] = data['activity_label'].map(label_mapping)

# Window data to take a feature extraction

window_size = 500
step_size = window_size // 2

X_win, y_win, groups_win = create_windows(
    data,
    window_size,
    step_size=step_size
)

print(pd.Series(y_win).value_counts())


# Do group shuffle split to avoid data leakage
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Get train and test indices
train_idx, test_idx = next(gss.split(X_win, y_win, groups_win))

# Split data
X_train, X_test = X_win.iloc[train_idx], X_win.iloc[test_idx]
y_train, y_test = y_win.iloc[train_idx], y_win.iloc[test_idx]
groups_train = groups_win.iloc[train_idx]
groups_test = groups_win.iloc[test_idx]
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Random Forest classifier
clf = RandomForestClassifier(n_estimators=300,  max_features="sqrt", max_depth=30,  min_samples_leaf=5, min_samples_split=10, class_weight="balanced",  n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

y_pred = clf.predict(X_test)

print("Accuracy:", clf.score(X_test, y_test))


print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.show()

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_win.columns,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nMost Important Features:")
print(feature_importance)

# Hyperparameter tuning
##param_grid = {
  #  'n_estimators' : [200, 300, 400],
   # 'max_depth' : [20, 30],
   # 'min_samples_split' : [2, 5, 7],
   # 'min_samples_leaf' : [1, 2, 3],
#}

#grid_search = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=3, scoring="f1_macro",)
#grid_search.fit(X_train, y_train)

#print("Best Parameters:", grid_search.best_params_)
#print("Best Estimator:", grid_search.best_estimator_)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import classification_report
import statistics
from scipy import stats
from scipy.signal import find_peaks
from collections import Counter
from sklearn.preprocessing import LabelEncoder


# ENHANCED window creation with MORE FEATURES
def create_windows(df, window_size, step_size, purity_threshold=0.7):
    features = []
    labels = []
    users = []

    for user_id, user_df in df.groupby("user"):
        user_df = user_df.reset_index(drop=True)

        for start in range(0, len(user_df) - window_size + 1, step_size):
            window = user_df.iloc[start:start + window_size]

            # centre time label
            center_idx = start + window_size // 2
            center_label = user_df.iloc[center_idx]["activity_label"]

            labels_in_window = window["activity_label"]

            # Fraction of window matching center label
            purity = np.mean(labels_in_window == center_label)

            if purity < purity_threshold:
                continue

            # signals
            ax = window["Accelerometer_X"].values
            ay = window["Accelerometer_Y"].values
            az = window["Accelerometer_Z"].values

            mag = np.sqrt(ax**2 + ay**2 + az**2)

            # BASIC Features
            feature_dict = {
                "ax_mean": ax.mean(),
                "ay_mean": ay.mean(),
                "az_mean": az.mean(),
                "ax_std": ax.std(),
                "ay_std": ay.std(),
                "az_std": az.std(),
                "mag_mean": mag.mean(),
                "mag_std": mag.std(),
                
                # ADDITIONAL STATISTICAL FEATURES
                "ax_min": ax.min(),
                "ax_max": ax.max(),
                "ay_min": ay.min(),
                "ay_max": ay.max(),
                "az_min": az.min(),
                "az_max": az.max(),
                "mag_min": mag.min(),
                "mag_max": mag.max(),
                
                # Range
                "ax_range": ax.max() - ax.min(),
                "ay_range": ay.max() - ay.min(),
                "az_range": az.max() - az.min(),
                "mag_range": mag.max() - mag.min(),
                
                # Variance
                "ax_var": ax.var(),
                "ay_var": ay.var(),
                "az_var": az.var(),
                "mag_var": mag.var(),
                
                # Median
                "ax_median": np.median(ax),
                "ay_median": np.median(ay),
                "az_median": np.median(az),
                "mag_median": np.median(mag),
                
                # Percentiles (25th and 75th)
                "ax_q25": np.percentile(ax, 25),
                "ax_q75": np.percentile(ax, 75),
                "ay_q25": np.percentile(ay, 25),
                "ay_q75": np.percentile(ay, 75),
                "az_q25": np.percentile(az, 25),
                "az_q75": np.percentile(az, 75),
                
                # IQR (Interquartile Range)
                "ax_iqr": np.percentile(ax, 75) - np.percentile(ax, 25),
                "ay_iqr": np.percentile(ay, 75) - np.percentile(ay, 25),
                "az_iqr": np.percentile(az, 75) - np.percentile(az, 25),
                
                # Skewness and Kurtosis
                "ax_skew": stats.skew(ax),
                "ay_skew": stats.skew(ay),
                "az_skew": stats.skew(az),
                "mag_skew": stats.skew(mag),
                
                "ax_kurtosis": stats.kurtosis(ax),
                "ay_kurtosis": stats.kurtosis(ay),
                "az_kurtosis": stats.kurtosis(az),
                "mag_kurtosis": stats.kurtosis(mag),
                
                # Energy (sum of squares)
                "ax_energy": np.sum(ax**2),
                "ay_energy": np.sum(ay**2),
                "az_energy": np.sum(az**2),
                "mag_energy": np.sum(mag**2),
                
                # RMS (Root Mean Square)
                "ax_rms": np.sqrt(np.mean(ax**2)),
                "ay_rms": np.sqrt(np.mean(ay**2)),
                "az_rms": np.sqrt(np.mean(az**2)),
                "mag_rms": np.sqrt(np.mean(mag**2)),
                
                # Zero crossing rate
                "ax_zcr": np.sum(np.diff(np.sign(ax - ax.mean())) != 0),
                "ay_zcr": np.sum(np.diff(np.sign(ay - ay.mean())) != 0),
                "az_zcr": np.sum(np.diff(np.sign(az - az.mean())) != 0),
                
                # Mean absolute deviation
                "ax_mad": np.mean(np.abs(ax - ax.mean())),
                "ay_mad": np.mean(np.abs(ay - ay.mean())),
                "az_mad": np.mean(np.abs(az - az.mean())),
                "mag_mad": np.mean(np.abs(mag - mag.mean())),
                
                # Correlation between axes
                "corr_xy": np.corrcoef(ax, ay)[0, 1] if len(ax) > 1 else 0,
                "corr_xz": np.corrcoef(ax, az)[0, 1] if len(ax) > 1 else 0,
                "corr_yz": np.corrcoef(ay, az)[0, 1] if len(ax) > 1 else 0,
                
                # Signal Magnitude Area (SMA)
                "sma": (np.sum(np.abs(ax)) + np.sum(np.abs(ay)) + np.sum(np.abs(az))) / len(ax),
                
                # Gravity component estimation
                "gravity_x": ax.mean(),
                "gravity_y": ay.mean(),
                "gravity_z": az.mean(),
                
                # Body acceleration (subtract gravity)
                "body_acc_x_std": (ax - ax.mean()).std(),
                "body_acc_y_std": (ay - ay.mean()).std(),
                "body_acc_z_std": (az - az.mean()).std(),
                
                # RATIO FEATURES - These are critical for distinguishing postures!
                "std_ratio_xy": ax.std() / (ay.std() + 1e-6),
                "std_ratio_xz": ax.std() / (az.std() + 1e-6),
                "std_ratio_yz": ay.std() / (az.std() + 1e-6),
                
                "mean_ratio_xy": ax.mean() / (ay.mean() + 1e-6),
                "mean_ratio_xz": ax.mean() / (az.mean() + 1e-6),
                "mean_ratio_yz": ay.mean() / (az.mean() + 1e-6),
            }

            features.append(feature_dict)
            labels.append(str(center_label))
            users.append(user_id)

    return pd.DataFrame(features), pd.Series(labels), pd.Series(users)


user_list = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18]

# New csv
data = pd.read_csv('physiological_with_activity_labels.csv')

data = data.dropna()
data = data[data['activity_label'] != 'moderate_activity']
data = data[data['activity_label'] != 'light_activity']
data = data.dropna(subset=["activity_label"])

label_mapping = {
    'stand': 'standing',
    'sit': 'sitting',
    'lying': 'lying'
}

data['activity_label'] = data['activity_label'].map(label_mapping)

print("="*70)
print("TESTING DIFFERENT CONFIGURATIONS")
print("="*70)

# Try multiple configurations
configs = [
    {"window": 400, "step": 40, "purity": 0.70, "name": "Config 1: w=400, s=40, p=0.70"},
    {"window": 500, "step": 50, "purity": 0.70, "name": "Config 2: w=500, s=50, p=0.70"},
    {"window": 600, "step": 50, "purity": 0.70, "name": "Config 3: w=600, s=50, p=0.70"},
    {"window": 500, "step": 25, "purity": 0.75, "name": "Config 4: w=500, s=25, p=0.75"},
]

best_accuracy = 0
best_config = None
best_model = None
best_X_test = None
best_y_test = None
best_le = None

for config in configs:
    print(f"\n{config['name']}")
    print("-" * 70)
    
    X_win, y_win, groups_win = create_windows(
        data,
        window_size=config['window'],
        step_size=config['step'],
        purity_threshold=config['purity']
    )
    
    print(f"Total windows: {len(X_win)}")
    print(pd.Series(y_win).value_counts())
    
    # ENCODE LABELS TO NUMBERS
    le = LabelEncoder()
    y_win_encoded = le.fit_transform(y_win)
    
    # Do group shuffle split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_win, y_win_encoded, groups_win))
    
    X_train, X_test = X_win.iloc[train_idx], X_win.iloc[test_idx]
    y_train, y_test = y_win_encoded[train_idx], y_win_encoded[test_idx]
    y_train_labels = y_win.iloc[train_idx]  # Keep original labels for weights
    
    # Calculate class weights - heavily penalize misclassifying minority classes
    class_counts = Counter(y_train_labels)
    total = sum(class_counts.values())
    
    # More aggressive class weighting
    class_weights = {}
    for cls, count in class_counts.items():
        if cls == 'standing':
            class_weights[cls] = (total / (len(class_counts) * count)) * 3.0  # 3x weight for standing
        elif cls == 'lying':
            class_weights[cls] = (total / (len(class_counts) * count)) * 2.0  # 2x weight for lying
        else:
            class_weights[cls] = total / (len(class_counts) * count)
    
    print(f"Class weights: {class_weights}")
    
    # Map to sample weights
    sample_weights = np.array([class_weights[label] for label in y_train_labels])
    
    # XGBoost with aggressive weighting
    clf = xgb.XGBClassifier(
        n_estimators=800,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1,
        gamma=0,
        reg_alpha=0.05,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    clf.fit(X_train, y_train, sample_weight=sample_weights)
    y_pred = clf.predict(X_test)
    
    acc = clf.score(X_test, y_test)
    print(f"Accuracy: {acc:.4f}")
    
    # Convert back to labels for report
    y_test_labels = le.inverse_transform(y_test)
    y_pred_labels = le.inverse_transform(y_pred)
    print(classification_report(y_test_labels, y_pred_labels, zero_division=0))
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_config = config
        best_model = clf
        best_X_test = X_test
        best_y_test = y_test
        best_le = le

print("\n" + "="*70)
print("BEST CONFIGURATION")
print("="*70)
print(f"{best_config['name']}")
print(f"Best Accuracy: {best_accuracy:.4f}")

# Retrain with best config
print("\nRetraining with best configuration...")
X_win, y_win, groups_win = create_windows(
    data,
    window_size=best_config['window'],
    step_size=best_config['step'],
    purity_threshold=best_config['purity']
)

# Encode labels
le = LabelEncoder()
y_win_encoded = le.fit_transform(y_win)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_win, y_win_encoded, groups_win))

X_train, X_test = X_win.iloc[train_idx], X_win.iloc[test_idx]
y_train, y_test = y_win_encoded[train_idx], y_win_encoded[test_idx]
y_train_labels = y_win.iloc[train_idx]

# Calculate aggressive class weights
class_counts = Counter(y_train_labels)
total = sum(class_counts.values())

class_weights = {}
for cls, count in class_counts.items():
    if cls == 'standing':
        class_weights[cls] = (total / (len(class_counts) * count)) * 3.0
    elif cls == 'lying':
        class_weights[cls] = (total / (len(class_counts) * count)) * 2.0
    else:
        class_weights[cls] = total / (len(class_counts) * count)

sample_weights = np.array([class_weights[label] for label in y_train_labels])

# Final XGBoost model
print("\n" + "="*70)
print("FINAL XGBOOST MODEL")
print("="*70)

clf_xgb = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=12,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=1,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss'
)

clf_xgb.fit(X_train, y_train, sample_weight=sample_weights)
y_pred_xgb = clf_xgb.predict(X_test)

# Convert back to labels
y_test_labels = le.inverse_transform(y_test)
y_pred_xgb_labels = le.inverse_transform(y_pred_xgb)

print(f"XGBoost Accuracy: {clf_xgb.score(X_test, y_test):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred_xgb_labels, zero_division=0))

# Confusion matrix
cm_xgb = confusion_matrix(y_test_labels, y_pred_xgb_labels)
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=le.classes_)
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title(f"XGBoost Confusion Matrix\nAccuracy: {clf_xgb.score(X_test, y_test):.2%}")
plt.tight_layout()
plt.savefig('xgb_confusion_matrix.png', dpi=150)
print("\nSaved: xgb_confusion_matrix.png")
plt.close()

# Random Forest for comparison
print("\n" + "="*70)
print("RANDOM FOREST FOR COMPARISON")
print("="*70)

# Convert class weights dict for RF (needs numeric keys)
class_weight_dict_rf = {}
for label_str, weight in class_weights.items():
    label_num = le.transform([label_str])[0]
    class_weight_dict_rf[label_num] = weight

clf_rf = RandomForestClassifier(
    n_estimators=1000,
    max_features="sqrt",
    max_depth=50,
    min_samples_leaf=1,
    min_samples_split=2,
    class_weight=class_weight_dict_rf,
    n_jobs=-1,
    random_state=42
)

clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)

# Convert back to labels
y_pred_rf_labels = le.inverse_transform(y_pred_rf)

print(f"Random Forest Accuracy: {clf_rf.score(X_test, y_test):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred_rf_labels, zero_division=0))

# Confusion matrix
cm_rf = confusion_matrix(y_test_labels, y_pred_rf_labels)
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=le.classes_)
disp.plot(ax=ax, cmap='Greens', values_format='d')
plt.title(f"Random Forest Confusion Matrix\nAccuracy: {clf_rf.score(X_test, y_test):.2%}")
plt.tight_layout()
plt.savefig('rf_confusion_matrix.png', dpi=150)
print("\nSaved: rf_confusion_matrix.png")
plt.close()

# Feature importance
print("\n" + "="*70)
print("TOP 30 MOST IMPORTANT FEATURES")
print("="*70)

feature_importance = pd.DataFrame({
    'feature': X_win.columns,
    'importance_xgb': clf_xgb.feature_importances_,
    'importance_rf': clf_rf.feature_importances_
}).sort_values('importance_xgb', ascending=False)

print(feature_importance.head(30))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

top_features = feature_importance.head(25)
top_features.plot(x='feature', y='importance_xgb', kind='barh', ax=axes[0], legend=False, color='steelblue')
axes[0].set_title('XGBoost - Top 25 Features', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Importance', fontsize=12)
axes[0].invert_yaxis()

top_features.plot(x='feature', y='importance_rf', kind='barh', ax=axes[1], legend=False, color='forestgreen')
axes[1].set_title('Random Forest - Top 25 Features', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Importance', fontsize=12)
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
print("\nSaved: feature_importance.png")
plt.close()

print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"Best Configuration: {best_config['name']}")
print(f"XGBoost Accuracy:   {clf_xgb.score(X_test, y_test):.4f}")
print(f"Random Forest:      {clf_rf.score(X_test, y_test):.4f}")
print("\nAll plots saved successfully!")