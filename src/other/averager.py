import pandas as pd
from random import uniform
import math
f = open("C:/Users/seanm/Desktop/GE_Mammography_Classification/results/oversampled_test_1/16-29-27.txt", "r")
totalTP = 0
totalAUC = 0
counter = 0
for line in f:
    if "Test TPR" in line:
        split = line.split(": ")
        number = float(split[1])
        counter += 1
        totalTP += number
    if "Test AUC" in line:
        split = line.split(": ")
        number = float(split[1])
        totalAUC += number
f.close()
print("Average TPR: ", totalTP / counter)
print("Average AUC: ", totalAUC / counter)