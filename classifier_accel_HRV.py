import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# -----------------------------
# 1) Load and clean data
# -----------------------------
DATA_PATH = "physiological_with_activity_labels.csv"

data = pd.read_csv(DATA_PATH)

# Keep only the columns we need (reduces mixed dtype issues)
required_cols = [
    "user",
    "activity_label",
    "Accelerometer_X",
    "Accelerometer_Y",
    "Accelerometer_Z",
    "HRV",
]
missing = [c for c in required_cols if c not in data.columns]
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}")

data = data[required_cols].copy()

# Drop rows missing anything essential
data = data.dropna(subset=required_cols)

# Ensure types are numeric where needed
data["user"] = pd.to_numeric(data["user"], errors="coerce")
for col in ["Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z", "HRV"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.dropna(subset=["user", "Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z", "HRV"])
data["user"] = data["user"].astype(int)

# Remove moderate_activity 
data = data[data["activity_label"] != "moderate_activity"].copy()

# Map labels to consistent names
label_mapping = {
    "stand": "standing",
    "sit": "sitting",
    "light_activity": "light_activity",
    "lying": "lying",
}
data["activity_label"] = data["activity_label"].map(label_mapping)

# Drop any rows that became NaN due to mapping (unexpected labels)
data = data.dropna(subset=["activity_label"]).copy()

# Optional: clip extreme HRV outliers (helps stability if HRV has spikes)
low, high = data["HRV"].quantile([0.01, 0.99])
data["HRV"] = data["HRV"].clip(lower=low, upper=high)

print("Final dataset shape:", data.shape)
print("Label counts:\n", data["activity_label"].value_counts())
print("\nUsers in dataset:", sorted(data["user"].unique()))

# -----------------------------
# 2) Define features/labels/groups
# -----------------------------
X = data[["Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z", "HRV"]].copy()
y = data["activity_label"].copy()
groups = data["user"].copy()

print("\nFeature preview:\n", X.head())

# -----------------------------
# 3) Group-aware train/test split (prevents leakage)
# -----------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

print("\nTrain users:", sorted(groups_train.unique()))
print("Test users:", sorted(groups_test.unique()))

# -----------------------------
# 4) Train Random Forest
# -----------------------------
clf = RandomForestClassifier(
    n_estimators=200,
    max_features="log2",
    max_depth=30,
    min_samples_leaf=5,
    min_samples_split=7,
    n_jobs=-1,
    random_state=42
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\nAccuracy:", clf.score(X_test, y_test))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# -----------------------------
# 5) Confusion matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(values_format="d")
plt.title("Random Forest: Accelerometer + HRV")
plt.tight_layout()
plt.show()

# -----------------------------
# 6) Feature importance (useful for your report)
# -----------------------------
importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature importances:\n", importances)

importances.plot(kind="bar")
plt.title("Feature importance: Accelerometer + HRV")
plt.tight_layout()
plt.show()
