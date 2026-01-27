import pandas as pd
import re
import numpy as np

path = "Activity Diary.xlsx"
raw = pd.read_excel(path)

raw.head()

id_cols = ["SAMPLING ID (CAT number)", "Date"]  # keep these
# (Optional) keep Name/Email if you want:
for c in ["Name", "Email"]:
    if c in raw.columns:
        id_cols.append(c)

pattern = re.compile(r"activity\s*(\d+)", re.IGNORECASE)
activity_cols = [c for c in raw.columns if pattern.search(c)]

len(activity_cols), activity_cols[:6]

long = raw[id_cols + activity_cols].melt(
    id_vars=id_cols,
    var_name="attribute",
    value_name="value"
)

long["activity_number"] = long["attribute"].str.extract(r"activity\s*(\d+)", flags=re.IGNORECASE).astype(int)

a = long["attribute"].str.lower()

long["field"] = np.select(
    [
        a.str.startswith("describe activity"),
        a.str.startswith("what time of day did activity"),
        a.str.startswith("rate the intensity"),
        a.str.startswith("insert the number of minutes"),
    ],
    ["description", "time_of_day", "intensity", "minutes"],
    default="other"
)

long = long[long["field"] != "other"]

tidy = (
    long.pivot_table(
        index=id_cols + ["activity_number"],
        columns="field",
        values="value",
        aggfunc="first"
    )
    .reset_index()
)

tidy.columns.name = None
tidy.head()

tidy["Date"] = pd.to_datetime(tidy["Date"], errors="coerce")
tidy["intensity"] = pd.to_numeric(tidy["intensity"], errors="coerce")
tidy["minutes"] = pd.to_numeric(tidy["minutes"], errors="coerce")

# drop blank/filler rows
tidy["description"] = tidy["description"].astype(str).str.strip()
tidy = tidy[~tidy["description"].isin(["0", "0.0", "", "nan", "None"])]
tidy = tidy[(tidy["minutes"].fillna(0) > 0) | (tidy["intensity"].fillna(0) > 0) | tidy["time_of_day"].notna()]
tidy = tidy.reset_index(drop=True)

t = pd.to_datetime(tidy["time_of_day"].astype(str), format="%H:%M", errors="coerce")

tidy["activity_datetime"] = tidy["Date"] + pd.to_timedelta(t.dt.hour, unit="h") + pd.to_timedelta(t.dt.minute, unit="m")

tidy = tidy.sort_values(["SAMPLING ID (CAT number)", "Date", "activity_datetime"]).reset_index(drop=True)

def fix_midnight_wrap(group):
    group = group.sort_values("activity_datetime").copy()
    # we want original order by activity_number, not sorted by time, so:
    group = group.sort_values("activity_number").copy()

    add_days = 0
    fixed = []
    prev = None

    for dt_val in group["activity_datetime"]:
        if pd.isna(dt_val):
            fixed.append(pd.NaT)
            continue
        if prev is not None and dt_val < prev:
            add_days += 1
        fixed.append(dt_val + pd.Timedelta(days=add_days))
        prev = dt_val + pd.Timedelta(days=add_days)

    group["activity_datetime"] = fixed
    return group

tidy = tidy.groupby(["SAMPLING ID (CAT number)", "Date"], group_keys=False).apply(fix_midnight_wrap)

output_path = "updated_activity_labels.xlsx"
tidy.to_excel(output_path, index=False)
