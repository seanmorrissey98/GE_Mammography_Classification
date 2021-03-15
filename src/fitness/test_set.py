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
        n_points = len(self.test) - round(len(data) * .20) 
        
        for i in range(self.start, n_points):
            main = []
            opposite = []
            for j in range(52):
                main.append(data["x"+str(j)][i])
                opposite.append(data["x"+str(j+52)][i])

            # Append output of classifier to program output list
            progOuts.append(self.exec(main, opposite))
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
        self.getTruePositiveRate(progOuts)
        return self.getRocAucScore(progOuts, n_points)


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
                if self.labels[self.start+i] == 0:
                    # False Positive
                    fp = fp + 1
            else:  # Guessing suspicious area not present
                if self.labels[self.start+i] == 1:
                    # False Negative
                    fn = fn + 1
        return (fp + fn) / len(progOuts)

    def getTruePositiveRate(self, progOuts):
        tp, fn = 0, 0
        for i in range(len(progOuts)):
            guess = 0
            if progOuts[i] > self.boundary:  # Guessing suspicious area present
                guess = 1
                if self.labels[self.start+i] == 1:
                    # Correct guess increase true positive counter
                    tp = tp + 1
            else:  # Guessing suspicious area not present
                guess = 0
                if self.labels[self.start+i] == 1:
                    # Incorrect guess increase false negative counter
                    fn = fn + 1
            print("Guess=",guess)
            print("Actual=",self.labels[self.start+i])
        fn = 1 if tp + fn == 0 else fn
        print("True Positives: ", tp)
        print("False Negatives: ", fn)
        print("TP Rate: ", tp/(tp+fn))
        return tp/(tp+fn)

    def getRocAucScore(self, progOuts, n_points):
        predictions = []
        for i in range(len(progOuts)):
            if progOuts[i] > self.boundary:  # Guessing suspicious area present
                predictions.append(1)
            else:  # Guessing suspicious area not present
                predictions.append(0)
        print("AUC: ", roc_auc_score(self.labels[self.start:n_points], predictions))
        return roc_auc_score(self.labels[self.start:n_points], predictions)

    def exec(self, main, opposite):
        x = 0.0
        index = 50
        if main[index] < 1:
            if opposite[index] < opposite[index] + 0.00001:
                x = (x + 0.0001)
        else:
            if main[index] < main[index] + 7:
                x = (x + 0.00001)
            else:
                    x = (x - 0.7)
        index = 27
        if abs(sum(main) - sum(opposite)) > 5000:
                x = (x + 0.0001)
        index = 21
        if main[index] < 1:
            if opposite[index] > main[index] + 0.0000001:
                x = (x + 0.0000001)
        else:
            if main[index] > opposite[index] + 8:
                x = (x - 0.1)
            else:
                x = (x + 0.4)
        index = 15
        if main[index] < 1:
            if main[index] < main[index] + 0.5:
                x = (x - 0.00001)
        else:
            if main[index] < main[index] + 4:
                x = (x + 0.001)
            else:
                x = (x + 0.4)
        index = 4
        if abs(sum(main[-index:]) - sum(opposite[-index:])) > 100:
            x = (x - 0.9)
        index = 4
        if main[index] < 1:
            if opposite[index] > main[index] + 1.0:
                x = (x + 0.6)
        else:
            if main[index] < main[index] + 2:
                x = (x - 0.0000001)
            else:
                x = (x - 0.01)
        return x

t = test_set()
t.evaluate()