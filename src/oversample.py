import pandas as pd

in_file = "C:/Users/seanm/Desktop/GE_Mammography_Classification/data/haralick02_50K.csv"
df = pd.read_csv(in_file)
max_size = df['Label'].value_counts().max()
lst = [df]
for class_index, group in df.groupby('Label'):
    lst.append(group.sample(max_size-len(group), replace=True))
frame_new = pd.concat(lst)
frame_new = frame_new.sample(frac=1).reset_index(drop=True)
frame_new.to_csv('haralick_preparedV2.csv')