import pandas as pd
import re
import numpy as np

path = "Activity Diary.xlsx"
raw = pd.read_excel(path)

raw.head()

ID_COL = "SAMPLING ID (CAT number)"
DATE_COL = "Date"

pattern = re.compile(r"activity\s*(\d+)", re.IGNORECASE)
activity_cols = [c for c in raw.columns if pattern.search(c)]

long = raw[[ID_COL, DATE_COL] + activity_cols].melt(
    id_vars=[ID_COL, DATE_COL],
    var_name="attribute",
    value_name="value"
)

long["activity_number"] = (
    long["attribute"]
    .str.extract(r"activity\s*(\d+)", flags=re.IGNORECASE)
    .astype(int)
)

a = long["attribute"].str.lower()

long["field"] = np.select(
    [
        a.str.startswith("describe activity"),
        a.str.startswith("rate the intensity"),
        a.str.startswith("insert the number of minutes"),
        a.str.startswith("what time of day did activity"),
    ],
    ["description", "intensity", "minutes", "time_of_day"],
    default=None
)

long = long.dropna(subset=["field"])

tidy = (
    long.pivot_table(
        index=[ID_COL, DATE_COL, "activity_number"],
        columns="field",
        values="value",
        aggfunc="first"
    )
    .reset_index()
)

tidy.columns.name = None

tidy = tidy.rename(columns={
    ID_COL: "sampling_id",
    DATE_COL: "date"
})

tidy.columns

tidy.to_excel("updated_activity_labels_2.xlsx", index=False)
