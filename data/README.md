# Machine Learning using Grammatical Evolution Project
--------------
## Datasets
Two primary datasets were used throughout this project which include:
- `haralick02_50K.csv`
- `haralick_preparedV2.csv`

Both datasets play a role in different use cases with further information on both below.
#### **`haralick02_50K.csv`**
This dataset is the original dataset used for this project and contains a total of 4999 segments of mammograms, with 217 positive class instances (cancerous / suspicious segments) and 4782 negative class instances. 

The dataset contains 104 columns of Haralick features per segment. 52 of those columns correspond to the segment on the left breast and the other 52 correspond to the same segment on the right breast. This is done as we can consider textural asymmetry across breasts as a potential indicator for suspicious areas. 

Haralick features describe textural features for an image such as variance, entropy and correlation. We can use these textural features for the comparison of the segment on each breast to find differences between the two.

#### **`haralick_preparedV2.csv`**
This dataset contains the same data as the original dataset however, it increases the number of positive class instances. The number of positive class instances was increased from 217 from the original dataset to 4782 in this dataset. The reason for this oversampling is to address the class imbalance problem. This dataset in total contains 4782 positive class instances and 4782 negative class instances, meaning 50% of the dataset is positive class instances while with the original dataset only 4.34% of the data was positive class instances.