import pandas as pd
from sklearn.metrics import roc_auc_score

class test_set():

    def __init__(self):
        in_file = "C:/Users/seanm/Desktop/GE_Mammography_Classification/data/haralick02_50K.csv"
        df = pd.read_csv(in_file)
        df = df.sample(frac=1).reset_index(drop=True)

        haralick_features = []
        for i in range(104):
            feature = "x"+ str(i)
            haralick_features.append(feature)

        self.data = df[haralick_features]
        self.labels = df['Label']
        self.test = self.data

    def evaluate(self):
        # ind.phenotype will be a string, including function definitions etc.
        # When we exec it, it will create a value XXX_output_XXX, but we exec
        # inside an empty dict for safety.

        data = self.test
        progOuts = []
        self.start = 0
        self.boundary = 0
        self.n_points = len(self.test) - round(len(data) * .20) 
        
        for i in range(self.start, self.n_points):
            main = []
            opposite = []
            for j in range(52):
                main.append(data["x"+str(j)][i])
                opposite.append(data["x"+str(j+52)][i])

            # Append output of classifier to program output list
            progOuts.append(self.exec(main, opposite))
        
        # Loop finished we now have all classifier output for each row in the training set
        # We now initialise all variables for OICB
        initMid = progOuts[round(len(progOuts) / 2)]
        max = progOuts[len(progOuts) - 1]
        min = progOuts[0]
        initMin = (initMid + min) / 2
        initMax = (initMid + max) / 2
        error = 1
        self.getBoundary(min, max, initMid, initMin, initMax, error, progOuts)
        self.getTruePositiveRate(progOuts)
        return self.getRocAucScore(progOuts)


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
        training_labels = self.labels[self.start:self.n_points].values.tolist()
        for i in range(len(progOuts)):
            if progOuts[i] > boundary:  # Guessing suspicious area present
                if training_labels[i] == 0:
                    # False Positive
                    fp = fp + 1
            else:  # Guessing suspicious area not present
                if training_labels[i] == 1:
                    # False Negative
                    fn = fn + 1
        return (fp + fn) / len(progOuts)

    def getRocAucScore(self, progOuts):
        predictions = []
        training_labels = self.labels[self.start:self.n_points].values.tolist()
        for i in range(len(progOuts)):
            if progOuts[i] > self.boundary:  # Guessing suspicious area present
                predictions.append(1)
            else:  # Guessing suspicious area not present
                predictions.append(0)
        print("AUC: ", roc_auc_score(training_labels, predictions))
        return roc_auc_score(training_labels, predictions)

    def getTruePositiveRate(self, progOuts):
        tp, fn = 0, 0
        tn, fp = 0, 0
        training_labels = self.labels[self.start:self.n_points].values.tolist()
        for i in range(len(progOuts)):
            if progOuts[i] > self.boundary:  # Guessing suspicious area present
                if training_labels[i] == 1:
                    # Correct guess increase true positive counter
                    tp = tp + 1
                else:
                    fp = fp + 1
            else:  # Guessing suspicious area not present
                if training_labels[i] == 1:
                    # Incorrect guess increase false negative counter
                    fn = fn + 1
                else:
                    tn = tn + 1
        fn = 1 if tp + fn == 0 else fn
        print("TP: ", tp/(tp+fn))
        return tp/(tp+fn)


    def exec(self, main, opposite):
        x = 0.0
        index = 39
        if main[index] < opposite[index]:
            x = (x - 0.001)
        index = 24
        if main[index] < 1:
            if main[index] < main[index] + 0.000001:
                x = (x - 0.001)
        else:
            if opposite[index] > main[index] + 1:
                x = (x - 0.7)
            else:
                x = (x - 0.9)
        index = 40
        if abs(sum(main) - sum(opposite)) > 1000:
            x = (x - 0.0001)
        index = 10
        if opposite[index] > sum(opposite[-index:]):
            x = (x - 0.8)
        index = 14
        if abs(sum(main) - sum(opposite)) > 10000:
            x = (x - 0.7)
        index = 7
        if main[index] < 1:
            if main[index] > main[index] + 0.8:
                x = (x - 0.1)
        else:
            if opposite[index] < opposite[index] + 2:
                x = (x - 0.0000001)
            else:
                x = (x + 0.8)
        index = 30
        if main[index] < 1:
            if opposite[index] < opposite[index] + 0.001:
                x = (x + 0.1)
        else:
            if main[index] < opposite[index] + 1:
                x = (x + 0.4)
            else:
                x = (x + 0.7)
        index = 26
        if opposite[index] > main[index]:
            x = (x + 0.001)
        index = 5
        if main[index] < 1:
            if main[index] < main[index] + 0.9:
                x = (x - 0.01)
        else:
            if opposite[index] < main[index] + 1:
                x = (x - 0.0000001)
            else:
                x = (x - 0.3)
        index = 42
        if sum(main) / 52 < main[index]:
            x = (x + 0.0000001)
        index = 2
        if opposite[index] > main[index]:
            x = (x - 0.001)
        index = 39
        if main[index] > main[index]:
            x = (x + 0.7)
        index = 30
        if sum(main) / 52 + sum(main) / 52 > opposite[index]:
            x = (x + 0.4)
        index = 3
        if main[index] < 1:
            if opposite[index] > opposite[index] + 0.7:
                x = (x - 0.4)
        else:
            if main[index] > opposite[index] + 1:
                x = (x - 0.2)
            else:
                x = (x - 0.8)
        index = 26
        if abs(sum(main[:-index]) - sum(opposite[-index:])) > 5000:
            x = (x - 0.4)
        index = 50
        if abs(sum(main[:-index]) - sum(opposite[:-index])) > 1000:
            x = (x - 0.00001)
        index = 34
        if opposite[index] > main[index]:
            x = (x - 0.001)
        index = 0
        if abs(sum(main) - sum(opposite)) > 5000:
            x = (x - 0.3)
        index = 46
        if sum(main) / 52 - opposite[index] < main[index] - main[index]:
            x = (x - 0.000001)
        index = 7
        if abs(sum(main) - sum(opposite)) > 1000:
            x = (x - 0.7)
        index = 46
        if main[index] > main[index]:
            x = (x - 0.4)
        index = 44
        if sum(main) / 52 > opposite[index]:
            x = (x + 0.000001)
        index = 0
        if sum(opposite) / 52 < sum(main) / 52:
            x = (x + 0.5)
        return x

t = test_set()
t.evaluate()