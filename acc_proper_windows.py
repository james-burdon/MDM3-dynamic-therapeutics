
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


user_list = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18]




# Some users are not in the activity log
users_to_remove = [19, 20, 21, 22]

# New csv
data = pd.read_csv('physiological_with_activity_labels.csv')

data = data.dropna()
data = data[data['activity_label'] != 'moderate_activity']

label_mapping = {
    'stand' : 'standing',
    'sit' : 'sitting',
    'light_activity' : 'light_activity',
    'lying' : 'lying'
}

data['activity_label'] = data['activity_label'].map(label_mapping)


# Extract accelerometer data
X_axs = data['Accelerometer_X']
Y_axs = data['Accelerometer_Y']
Z_axs = data['Accelerometer_Z']

# Combine features
X = pd.concat([X_axs, Y_axs, Z_axs], axis=1)
print(X.head())

# Extract classification labels
y = data['activity_label']

groups = data['user']

# Do group shuffle split to avoid data leakage
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Get train and test indices
train_idx, test_idx = next(gss.split(X, y, groups))

# Split data
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = groups.iloc[train_idx]
groups_test = groups.iloc[test_idx]

# Random Forest classifier
clf = RandomForestClassifier(n_estimators=200,  max_features="log2", max_depth=30,  min_samples_leaf=5, min_samples_split=7, n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

y_pred = clf.predict(X_test)

print("Accuracy:", clf.score(X_test, y_test))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.show()
