import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, GroupKFold


# Reduce the 3 Inclinometer columns with 1 column containing the posture labels
def get_posture(row):

    if row["Inclinometer Standing"] == 1:
        return 'standing'

    if row["Inclinometer Sitting"] == 1:
        return 'sitting'

    if row["Inclinometer Lying"] == 1:
        return 'lying'

    else:
        return 'unknown'
    
all_users = []

# Import data from all the users and add to list
for i in range(1,23):

  #  df = pd.read_csv("DataPaper/user_1/Actigraph.csv")
    df = pd.read_csv(f"DataPaper/user_{i}/Actigraph.csv")


    # Remove rows where there is no posture recorded
    df = df[df["Inclinometer Off"] == 0]

    # Add in a column with the classified posture
    df["posture"] = df.apply(get_posture, axis = 1)

    # Add user number
    df["user_id"] = i

    all_users.append(df)

data_set = pd.concat(all_users, ignore_index=True)


train_set, test_set = train_test_split(data_set, test_size=0.2, stratify=data_set["user_id"])


'''
Try windowing data
'''

# Use 2 seconds for a window size
window_size = 2

X = []
y = []
groups = [] # User id per window

for user_id, user_df in data_set.groupby("user_id"):
    for i in range(0, len(data_set) - window_size +1, window_size):

        window = data_set.iloc[i:i + window_size]

        features = []
    
        for axis in ["Axis1", "Axis2", "Axis3", "HR", "Vector Magnitude"]:

            # Use the mean and standard deviation of the data as features
            features.append(window[axis].mean())
            features.append(window[axis].std())
    
        X.append(features)

        # Use the most common posture accross a window
        label = window["posture"].mode()[0]
        y.append(label)
        groups.append(user_id)


X = np.array(X)
y = np.array(y)
groups = np.array(groups)

X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(X, y, groups, test_size=0.2, stratify=groups)


# Random Forest classifier
rf = RandomForestClassifier(n_estimators=200,  max_features="log2", max_depth=20, min_samples_leaf=2, min_samples_split=5)

# Train classifier
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.show()

print(classification_report(y_test, y_pred))

# Perform 5-fold cross-validation
# This takes ages to run
gkf = GroupKFold(n_splits=3)
scores = cross_val_score(rf, X_train, y_train, cv=gkf, groups=groups_train)

print("Cross-Validation Scores:", scores)
print("Mean Accuracy,", scores.mean())