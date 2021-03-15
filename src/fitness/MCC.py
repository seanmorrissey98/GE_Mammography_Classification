from fitness.base_ff_classes.base_ff import base_ff
import pandas as pd
import math
from random import uniform

class MCC(base_ff):
    """
    derived from Py-max
    """
    maximise = True  # True as it ever was.

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()

        in_file = "C:/Users/seanm/Desktop/GE_Mammography_Classification/data/haralick02_50K.csv"
        df = pd.read_csv(in_file)
        df.sort_values(by=['Label'], inplace=True)
        df.to_csv('sortedMCC.csv')
        #max_size = df['Label'].value_counts().max()
        #lst = [df]
        #for class_index, group in df.groupby('Label'):
        #    lst.append(group.sample(max_size-len(group), replace=True))
        #df = pd.concat(lst)
        haralick_features = []
        for i in range(104):
            feature = "x"+ str(i)
            haralick_features.append(feature)
        self.data = df[haralick_features]
        self.labels = df['Label']
        self.training = self.data
        self.test = self.data
        self.n_vars = len(self.data)

        self.training_test = True
    
    def evaluate(self, ind, **kwargs):
        # ind.phenotype will be a string, including function definitions etc.
        # When we exec it, it will create a value XXX_output_XXX, but we exec
        # inside an empty dict for safety.

        dist = kwargs.get('dist', 'training')
        data = self.data
        progOuts = []
        self.start = 0
        self.boundary = 0
        

        p, d = ind.phenotype, {}
        self.points = self.getPIRS()
        for i in (self.points):
            main = []
            opposite = []
            for j in range(52):
                main.append(data["x"+str(j)][i])
                opposite.append(data["x"+str(j+52)][i])
            d["main"] = main
            d["opposite"] = opposite
            d['n_points'] = len(d['main'])

            exec(p, d)
            # Append output of classifier to program output list
            progOuts.append(d["XXX_output_XXX"])
            progOuts.sort()
        
        # Loop finished we now have all classifier output for each row in the training set
        # We now initialise all variables for OICB
        initMid = progOuts[round(len(progOuts) / 2)]
        max = progOuts[len(progOuts) - 1]
        min = progOuts[0]
        initMin = (initMid + min) / 2
        initMax = (initMid + max) / 2
        error = 1
        self.getBoundary(min, max, initMid, initMin, initMax, error, progOuts)
        return self.getMCC(progOuts)

    def getBoundary(self, lowerLimit, upperLimit, mid, bottom, top, errorCount, progOutput):
        # Calculate the classification error for mid, top and bottom boundaries
        midError = self.getClassificationErrors(mid, progOutput)
        botError = self.getClassificationErrors(bottom, progOutput)
        topError = self.getClassificationErrors(top, progOutput)

        # Boundary closer to middle
        if midError <= botError and midError <= topError:
            bestError = midError
            bestBoundary = mid
            newMid = mid
            newTop = (mid + top) / 2
            newBottom = (mid + bottom) / 2
        # Boundary closer to bottom
        elif botError <= midError and botError <= topError:
            bestError = botError
            bestBoundary = bottom
            newMid = bottom
            newTop = (mid + bottom) / 2
            newBottom = (lowerLimit + bottom) / 2
        # Boundary closer to the top
        else:
            bestError = topError
            bestBoundary = top
            newMid = top
            newTop = (mid + top) / 2
            newBottom = (upperLimit + top) / 2

        # Have not found the optimal boundary, try again
        if bestError < errorCount:
            errorCount = bestError
            self.boundary = bestBoundary
            self.getBoundary(lowerLimit, upperLimit, newMid, newBottom, newTop, errorCount, progOutput)
        else:
            # No better boundary to be found
            return

    """
    Loop through the program outputs comparing them to the boundary passed in.
    Calculate all false positives and false negatives and divide by the length of
    the program outputs to calculate the classification error for that specific
    boundary.
    """
    def getClassificationErrors(self, boundary, progOuts):
        fp, fn = 0, 0
        for i in range(len(progOuts)):
            if progOuts[i] > boundary:  # Guessing suspicious area present
                if self.labels[self.points[i]] == 0:
                    # False Positive
                    fp = fp + 1
            else:  # Guessing suspicious area not present
                if self.labels[self.points[i]] == 1:
                    # False Negative
                    fn = fn + 1
        return (fp + fn) / len(progOuts)

    def getMCC(self, progOuts):
        tp, fn = 0, 0
        tn, fp = 0, 0
        for i in range(len(progOuts)):
            if progOuts[i] > self.boundary:  # Guessing suspicious area present
                if self.labels[self.points[i]] == 1:
                    # Correct guess increase true positive counter
                    tp = tp + 1
                else:
                    fp = fp + 1
            else:  # Guessing suspicious area not present
                if self.labels[self.points[i]] == 1:
                    # Incorrect guess increase false negative counter
                    fn = fn + 1
                else:
                    tn = tn + 1
        numerator = ((tp * tn) - (fp * fn))
        denominator = math.sqrt((tp+fp)*(tp+tn)*(fp+fn)*(tn+fn))
        return numerator / denominator

    def getPIRS(self):
        benign = self.labels.value_counts()[0]
        malignant = self.labels.value_counts()[1]
        total = benign + malignant

        percentage_b = round(benign/total, 2)
        percentage_m = round(malignant/total, 2)

        percent_majority = round(uniform(percentage_m, percentage_b),2)
        percent_minority = round(1 - percent_majority,2)

        majority_datapoints = round(total * percent_majority)
        minority_datapoints = round(total * percent_minority)

        datapoints = []
        start = 0
        end = int(benign)+int(malignant)

        for i in range(int(majority_datapoints)):
            datapoints.append(round(uniform(start, int(benign))))

        for i in range(int(minority_datapoints)):
            datapoints.append(round(uniform(int(benign),end-1)))

        return datapoints