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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
from datetime import datetime, timedelta

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
day1_RR = []
day2_RR = []
hormone_diff = []
mean_diff = []
std_diff = []

all_users_2 = []
axis_1_day1 = []
axis_1_day2 = []
axis_2_day1 = []
axis_2_day2 = []
axis_3_day1 = []
axis_3_day2 = []
day1_axis1 = []
day2_axis1 = []
day1_axis2 = []
day2_axis2 = []
day1_axis3 = []
day2_axis3 = []
hours = []
minutes = []
cortisol_bb = []
# Import data from all the users and add to list
for i in range(1,23):
    if i==21:
        continue
    
    elif i == 11:
        continue

    else:
        df = pd.read_csv(f"DataPaper/user_{i}/Actigraph.csv")
        RR = pd.read_csv(f'DataPaper/user_{i}/RR.csv')
        hormone = pd.read_csv(f"DataPaper/user_{i}/saliva.csv")
        sleep = pd.read_csv(f"DataPaper/user_{i}/sleep.csv")

    
        bed_time = sleep["In Bed Time"][0]

        time = datetime.strptime(bed_time, "%H:%M")
        three_hours = timedelta(hours = 3)
        
        start = time - three_hours

    
                

        # Remove rows where there is no posture recorded
        df = df[df["Inclinometer Off"] == 0]


        # Add in a column with the classified posture
        df["posture"] = df.apply(get_posture, axis = 1)

        # Remove lying to try with only sitting and standing
    # df =df[df["posture"].isin(["sitting", "standing"])]

        # Add user number
        df["user_id"] = i

        diff_hormone = hormone['Cortisol NORM'].iloc[1] - hormone['Cortisol NORM'].iloc[0]
        cortisol_bb.append(hormone['Cortisol NORM'].iloc[0])
        day1_values = []
        day2_values = []
        for index, RR_row in RR.iterrows():
            if RR_row['day'] == 1:
                day1_values.append(RR_row['ibi_s'])
            elif RR_row['day'] == 2:
                day2_values.append(RR_row['ibi_s'])
        mean_RR_day1 = np.mean(day1_values) if day1_values else 0
        std_RR_day1 = np.std(day1_values) if day1_values else 0

        mean_RR_day2 = np.mean(day2_values) if day2_values else 0
        std_RR_day2 = np.std(day2_values) if day2_values else 0

        mean_diff.append(mean_RR_day1 - mean_RR_day2)
        std_diff.append(std_RR_day1 - std_RR_day2)

        all_users.append(df)
        day1_RR.append(mean_RR_day1)#, std_RR_day1))
        day2_RR.append(mean_RR_day2)#, std_RR_day2))
        hormone_diff.append(diff_hormone)
      

        for index, axis_row in df.iterrows():
            if axis_row['day'] == 1:
                if start < datetime.strptime(axis_row['time'], "%H:%M:%S") <= time:
                    axis_1_day1.append(axis_row['Axis1'])
                    axis_2_day1.append(axis_row['Axis2'])
                    axis_3_day1.append(axis_row['Axis3'])
            elif axis_row['day'] == 2:
                axis_1_day2.append(axis_row['Axis1'])
                axis_2_day2.append(axis_row['Axis2'])
                axis_3_day2.append(axis_row['Axis3'])

        mean_axis1_day1 = np.mean(axis_1_day1)
        std_axis1_day1 = np.std(axis_1_day1) 

        mean_axis1_day2 = np.mean(axis_1_day2) 
        std_axis1_day2 = np.std(axis_1_day2) 

        mean_axis2_day1 = np.mean(axis_2_day1)
        std_axis2_day1 = np.std(axis_2_day1) 

        mean_axis2_day2 = np.mean(axis_2_day2) 
        std_axis2_day2 = np.std(axis_2_day2) 

        mean_axis3_day1 = np.mean(axis_3_day1)
        std_axis3_day1 = np.std(axis_3_day1) 

        mean_axis3_day2 = np.mean(axis_3_day2) 
        std_axis3_day2 = np.std(axis_3_day2) 

        mean_diff.append(mean_RR_day1 - mean_RR_day2)
        std_diff.append(std_RR_day1 - std_RR_day2)

        day1_axis1.append(mean_axis1_day1)#, std_RR_day1))
        day2_axis1.append(mean_axis1_day2)#, std_RR_day2))

        day1_axis2.append(mean_axis2_day1)#, std_RR_day1))
        day2_axis2.append(mean_axis2_day2)#, std_RR_day2))

        day1_axis3.append(mean_axis3_day1)#, std_RR_day1))
        day2_axis3.append(mean_axis3_day2)#, std_RR_day2))
    
#data_set = pd.concat(all_users, ignore_index=True)
#day1_dataset = pd.concat([pd.DataFrame(day1_RR, columns=['mean_RR_day1', 'std_RR_day1'])], axis=1)
##day2_dataset = pd.concat([pd.DataFrame(day2_RR, columns=['mean_RR_day2', 'std_RR_day2'])], axis=1)
#hormone_dataset = pd.concat([pd.DataFrame(hormone_diff, columns=['hormone_diff'])], axis=1)
#features_ds = pd.concat([pd.DataFrame(mean_diff, columns=['mean_diff']), pd.DataFrame(std_diff, columns=['std_diff'])], axis=1) 

X = np.column_stack([
    day1_axis1,
    day2_axis1,
    day1_axis2,
    day2_axis2,
    day1_axis3,
    day2_axis3
])
# Test with only data three hours before sleep 
X_2 = np.column_stack([
    day1_axis1,
    day1_axis2,
    day1_axis3

])

#train, test, train_hormone, test_hormone = train_test_split(X, hormone_diff, test_size=0.2)
train, test, train_hormone, test_hormone = train_test_split(X_2[2:,], cortisol_bb[2:], test_size=0.2)
acc_regressor = LinearRegression(n_jobs=-1)
acc_regressor.fit(train, train_hormone)
predictions = acc_regressor.predict(test)
mse = mean_squared_error(test_hormone, predictions)
r2 = r2_score(test_hormone, predictions)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")


# Day before
r_before, p_before = spearmanr(day1_RR, hormone_diff)
print(r_before)
print(p_before)

# Day after  
r_after, p_after = spearmanr(day2_RR, hormone_diff)
print(r_after)
print(p_after)

#combined_users = pd.concat([day1_dataset, day2_dataset], axis=1)
print(len(hormone_diff))
plt.figure()
plt.scatter(day1_RR, hormone_diff)
plt.show()
plt.figure()
plt.scatter(day2_RR, hormone_diff)
plt.show()
    
med_hormone_change = np.median(hormone_diff)

#low_hormone_changes = [change if change < med_hormone_change for change in hormone_diff else None]
#high_hormone_changes = [change if change >= med_hormone_change for change in hormone_diff else None]

#train_rr, test_rr, train_hormone, test_hormone = train_test_split(features_ds, hormone_dataset, test_size=0.2)

#lr = RandomForestRegressor(n_jobs=-1)
#lr.fit(train_rr, train_hormone)
#predictions = lr.predict(test_rr)
#mse = mean_squared_error(test_hormone, predictions)
#r2 = r2_score(test_hormone, predictions)
#print(f"Mean Squared Error: {mse}")
#print(f"R^2 Score: {r2}")



