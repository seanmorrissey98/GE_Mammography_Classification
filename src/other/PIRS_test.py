import pandas as pd
from random import uniform
import math
in_file = "C:/Users/seanm/Desktop/GE_Mammography_Classification/data/haralick02_50K.csv"
df = pd.read_csv(in_file)
df.sort_values(by=['Label'], inplace=True)
#df.to_csv('sorted.csv')
benign = df['Label'].value_counts()[0]
malignant = df['Label'].value_counts()[1]
print("Benign: ", benign)
print("Malignant: ", malignant)
total = benign + malignant
percentage_b = round(benign/total, 2)
percentage_m = round(malignant/total, 2)
print("Benign % = ", percentage_b)
print("Malignant % = ", percentage_m)
percent_majority = round(uniform(percentage_m, percentage_b),2)
percent_minority = round(1 - percent_majority,2)
print("Percent Majority % = ", percent_majority)
print("Percent Minority % = ", percent_minority)
majority_datapoints = round(total * percent_majority)
minority_datapoints = round(total * percent_minority)
print("Total datapoints: ", total)
print("Majority/Benign datapoints: ", majority_datapoints)
print("Minority/Malignant datapoints: ", minority_datapoints)
datapoints = []
start = 0
end = int(benign)+int(malignant)
for i in range(int(majority_datapoints)):
    datapoints.append(round(uniform(start, int(benign))))
    #print(datapoints[i])
for i in range(int(minority_datapoints)):
    datapoints.append(round(uniform(int(benign),end)))
    #print(datapoints[len(datapoints)-1])
haralick_features = []
for i in range(104):
    feature = "x"+ str(i)
    haralick_features.append(feature)
data = df[haralick_features]
n_points = round(len(data) * .20)
points = list(range(0, n_points))
labels = df['Label']
Labels = labels[0:n_points].values.tolist()
print("Labels 1: ", len(points))
print("Labels 2: ", len(Labels))