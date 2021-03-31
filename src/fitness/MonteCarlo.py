from itertools import count
from typing import Counter
from fitness.base_ff_classes.base_ff import base_ff
import pandas as pd
from sklearn.metrics import roc_auc_score
import math
from random import uniform
import numpy as np

class MonteCarlo(base_ff):
    """
    An example of a single fitness class that generates
    two fitness values for multiobjective optimisation
    """

    maximise = True
    multi_objective = True
    default_fitness = [-1, -1]

    def __init__(self):

        # Initialise base fitness function class.
        super().__init__()
        self.num_obj = 2
        dummyfit = base_ff()
        dummyfit.maximise = True
        self.fitness_functions = [dummyfit, dummyfit]
        self.default_fitness = [-1, -1]

        in_file = "C:/Users/seanm/Desktop/GE_Mammography_Classification/data/haralick02_50K.csv"
        df = pd.read_csv(in_file)
        df.sort_values(by=['Label'], inplace=True)
        df.to_csv('sortedMCC.csv')

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
        self.counter = 0
        #self.first = True
        #self.first2 = True
        #self.tp_ind = []
        #self.auc_ind = []
        #self.avga_ind = []
        #self.mcc_ind = []

    def evaluate(self, ind, **kwargs):
        dist = kwargs.get('dist', 'training')
        data = []
        progOuts = []
        self.start = 0
        self.boundary = 0

        if dist == "training":
            # Set training datasets.
            data = self.training
            self.start = round(len(data) * .20)
            self.n_points = len(data)

        elif dist == "test":
            # Set test datasets.
            data = self.test
            self.start = 0
            self.n_points = round(len(data) * .20)
        p, d = ind.phenotype, {}
        training_attributes = data#[self.start:self.n_points]
        #training_labels = self.labels[self.start:self.n_points].values.tolist()
        self.points = self.getPIRS()
        for i in (self.points):
            main = []
            opposite = []
            for j in range(52):
                main.append(training_attributes["x"+str(j)][i])
                opposite.append(training_attributes["x"+str(j+52)][i])
            d["main"] = main
            d["opposite"] = opposite
            d['n_points'] = len(d['main'])

            exec(p, d)
            # Append output of classifier to program output list
            progOuts.append(d["XXX_output_XXX"])
            #progOuts.sort()
        # Loop finished we now have all classifier output for each row in the training set
        # We now initialise all variables for OICB
        initMid = progOuts[round(len(progOuts) / 2)]
        max = progOuts[len(progOuts) - 1]
        min = progOuts[0]
        initMin = (initMid + min) / 2
        initMax = (initMid + max) / 2
        error = 1
        self.getBoundary(min, max, initMid, initMin, initMax, error, progOuts)
        fitness = [self.getTruePositiveRate(progOuts), self.getRocAucScore(progOuts)]
        #self.tp_ind.append(self.getTruePositiveRate(progOuts))
        #self.auc_ind.append(self.getRocAucScore(progOuts))
        #self.avga_ind.append(self.getAVGA(progOuts))
        #self.mcc_ind.append(self.getMCC(progOuts))
        # self.counter = self.counter + 1
        # print(self.counter)
        #if self.counter == 50:
        #    self.monteCarlo(self.tp_ind, "TP")
        #    self.monteCarlo(self.auc_ind, "AUC")
        #    self.monteCarlo(self.avga_ind, "AVGA")
        #    self.monteCarlo(self.mcc_ind, "MCC")
        return fitness

    @staticmethod
    def value(fitness_vector, objective_index):
        """
        This is a static method required by NSGA-II for sorting populations
        based on a given fitness function, or for returning a given index of a
        population based on a given fitness function.

        :param fitness_vector: A vector/list of fitnesses.
        :param objective_index: The index of the desired fitness.
        :return: The fitness at the objective index of the fitness vecror.
        """
        if not isinstance(fitness_vector, list):
            return float("inf")

        return fitness_vector[objective_index]

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
        training_labels = self.correctLabels#[self.start:self.n_points].values.tolist()
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
        training_labels = self.correctLabels#[self.start:self.n_points].values.tolist()
        for i in range(len(progOuts)):
            if progOuts[i] > self.boundary:  # Guessing suspicious area present
                predictions.append(1)
            else:  # Guessing suspicious area not present
                predictions.append(0)
        return roc_auc_score(training_labels, predictions)

    def getTruePositiveRate(self, progOuts):
        tp, fn = 0, 0
        tn, fp = 0, 0
        training_labels = self.correctLabels#[self.start:self.n_points].values.tolist()
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
        return tp/(tp+fn)

    def getAVGA(self, progOuts):
        tp, fn = 0, 0
        tn, fp = 0, 0
        training_labels = self.correctLabels#[self.start:self.n_points].values.tolist()
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
        return 0.5 * (tp/(tp+fn) + tn/(tn+fp))

    def getMCC(self, progOuts):
        tp, fn = 0, 0
        tn, fp = 0, 0
        training_labels = self.correctLabels#[self.start:self.n_points].values.tolist()
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
        numerator = ((tp * tn) - (fp * fn))
        denominator = math.sqrt((tp+fp)*(tp+tn)*(fp+fn)*(tn+fn))
        return numerator / denominator

    def writeToFile(self, predictions, message, tofile):
        file = open(tofile, "a")
        file.write("Boundary = " + str(self.boundary)+"\n")
        file.write(str(message) + "\n")
        for i in range(len(predictions)):
            file.write("Actual: " + str(self.labels[self.start + i])+ " vs Predicted: " + str(predictions[i])+"\n")
        file.write("\n\n\n")
        file.close()

    def monteCarlo(self, population, text):
        file = open("MonteCarlo.txt", "a")
        average = self.getAverage(population)
        variance = self.getVariance(population, average)
        standardDeviation = self.getSDeviation(variance)
        file.write(text + " variance: " + str(variance) + "\n")
        file.write(text + " standard deviation: " + str(standardDeviation) + "\n")
        file.write(text + " average: " + str(average) + "\n\n")
        return

    def getVariance(self, population, average):
        sum = 0
        for i in population:
            sum = sum + ((i - average) * (i - average))
        return sum / len(population)

    def getSDeviation(self, variance):
        return math.sqrt(variance)

    def getAverage(self, population):
        sum = 0
        for i in population:
            sum = sum + i
        return sum / len(population)

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

        if majority_datapoints + minority_datapoints == 5000:
            majority_datapoints = majority_datapoints -1

        datapoints = []
        start = 0
        end = int(benign)+int(malignant)

        for i in range(int(majority_datapoints)):
            datapoints.append(round(uniform(start, int(benign))))

        for i in range(int(minority_datapoints)):
            datapoints.append(round(uniform(int(benign),end-1)))

        self.correctLabels = []
        for i in datapoints:
            # print("I: " + str(i) + "vs label: " + str(self.labels[i]))
            self.correctLabels.append(self.labels[i])
        return datapoints