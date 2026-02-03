
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



