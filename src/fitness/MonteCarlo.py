import math
import time
from itertools import count
from random import uniform
from typing import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fitness.base_ff_classes.base_ff import base_ff


class MonteCarlo(base_ff):
    """
    An example of a single fitness class that generates
    two fitness values for multiobjective optimisation
    """

    maximise = True
    multi_objective = True
    default_fitness = [-1, -1]

    def __init__(self):
        """Initialise base fitness function class and its variables.
        """
        super().__init__()
        self.num_obj = 2
        dummyfit = base_ff()
        dummyfit.maximise = True
        self.fitness_functions = [dummyfit, dummyfit]
        self.default_fitness = [-1, -1]
        t = time.localtime()
        current_time = time.strftime("%H-%M-%S", t)
        self.filename = current_time + ".txt"

        in_file = "../data/haralick02_50K.csv"
        df = pd.read_csv(in_file)

        haralick_features = []
        for i in range(104):
            feature = "x" + str(i)
            haralick_features.append(feature)
        self.data = df[haralick_features]
        self.labels = df['Label']
        self.training = self.data
        self.test = self.data

        self.counter = 0
        self.tp_ind = []
        self.auc_ind = []
        self.avga_ind = []
        self.mcc_ind = []

    def evaluate(self, ind, **kwargs):
        """Evaluate an individual on the test or training set using two fitness functions

        Args:
            ind (individual): A representation of the individual classifier

        Returns:
            list[float]: A list of the fitness values for the classifier
        """
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
            in_file = "../data/haralick02_50K.csv"
            df = pd.read_csv(in_file)
            self.labels = df['Label']

        p, d = ind.phenotype, {}
        training_attributes = data
        for i in range(self.start, self.n_points):
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

        # Loop finished we now have all classifier output for each row in the training set
        # We now initialise all variables for OICB
        initMid = progOuts[round(len(progOuts) / 2)]
        max = progOuts[len(progOuts) - 1]
        min = progOuts[0]
        initMin = (initMid + min) / 2
        initMax = (initMid + max) / 2
        error = 1
        self.getBoundary(min, max, initMid, initMin, initMax, error, progOuts)
        fitness = [self.getTruePositiveRate(
            progOuts), self.getRocAucScore(progOuts)]

        self.tp_ind.append(self.getTruePositiveRate(progOuts))
        self.auc_ind.append(self.getRocAucScore(progOuts))
        self.avga_ind.append(self.getAVGA(progOuts))
        self.mcc_ind.append(self.getMCC(progOuts))
        self.counter = self.counter + 1

        if self.counter == 10000:
            self.monteCarlo(self.tp_ind, "TP")
            self.monteCarlo(self.auc_ind, "AUC")
            self.monteCarlo(self.avga_ind, "AVGA")
            self.monteCarlo(self.mcc_ind, "MCC")

        return fitness

    @staticmethod
    def value(fitness_vector, objective_index):
        """
        This is a static method required by NSGA-II for sorting populations
        based on a given fitness function, or for returning a given index of a
        population based on a given fitness function.

        :param fitness_vector: A vector/list of fitnesses.
        :param objective_index: The index of the desired fitness.
        :return: The fitness at the objective index of the fitness vector.
        """
        if not isinstance(fitness_vector, list):
            return float("inf")

        return fitness_vector[objective_index]

    def getBoundary(self, lowerLimit, upperLimit, mid, bottom, top, errorCount, progOutput):
        """Sets the boundary to be used to be the best found by OICB

        Args:
            lowerLimit (float): The lowest value program output from the classifier
            upperLimit (float): The highest value program output from the classifier
            mid (float): The middle boundary to be tested
            bottom (float): The bottom boundary to be tested
            top (float): The top boundary to be tested
            errorCount (integer): The classification error
            progOutput (list[float]): A list of the program outputs from the classifier
        """
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
            self.getBoundary(lowerLimit, upperLimit, newMid,
                             newBottom, newTop, errorCount, progOutput)
        else:
            # No better boundary to be found
            return

    def getClassificationErrors(self, boundary, progOuts):
        """Returns the classification error for an individual based on a specific boundary

        Args:
            boundary (float): The boundary to use when calculating classification error
            progOuts (list[float]): A list of the program outputs from the classifier

        Returns:
            float: The classification error for an individual based on the boundary
        """
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
        """Gets area under the curve for a classifier based on its program outputs

        Args:
            progOuts (list[float]): A list of the program outputs from the classifier

        Returns:
            float: The AUC for a classifier
        """
        predictions = []
        training_labels = self.labels[self.start:self.n_points].values.tolist()
        for i in range(len(progOuts)):
            if progOuts[i] > self.boundary:  # Guessing suspicious area present
                predictions.append(1)
            else:  # Guessing suspicious area not present
                predictions.append(0)
        return roc_auc_score(training_labels, predictions)

    def getTruePositiveRate(self, progOuts):
        """Gets the true positive rate for a classifier based on its program outputs

        Args:
            progOuts (list[float]): A list of the program outputs from the classifier

        Returns:
            float: The true positive rate for a classifier
        """
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
        return tp/(tp+fn)

    def getFalsePositiveRate(self, progOuts):
        """Gets the false positive rate for a classifier based on its program outputs

        Args:
            progOuts (list[float]): A list of the program outputs from the classifier

        Returns:
            float: The false positive rate for a classifier
        """
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
        return -(fp/(fp+tn))

    def getAVGA(self, progOuts):
        """Gets the average accuracy for a classifier based on its program outputs

        Args:
            progOuts (list[float]): A list of the program outputs from the classifier

        Returns:
            float: The average accuracy for a classifier
        """
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
        return 0.5 * (tp/(tp+fn) + tn/(tn+fp))

    def getMCC(self, progOuts):
        """Gets the MCC score for a classifier based on its program outputs

        Args:
            progOuts (list[float]): A list of the program outputs from the classifier

        Returns:
            float: The Matthews Correlation Coefficient for a classifier
        """
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
        numerator = ((tp * tn) - (fp * fn))
        denominator = math.sqrt((tp+fp)*(tp+tn)*(fp+fn)*(tn+fn))
        return numerator / denominator

    def monteCarlo(self, population, text):
        """Monte Carlo Simulation which calculates metrics for a specific fitness function

        Args:
            population (list[float]): A list of the initial generations fitness values
            text (string): The name of the fitness function, used when writing to file
        """
        file = open("MonteCarlo.txt", "a")
        average = self.getAverage(population)
        variance = self.getVariance(population, average)
        standardDeviation = self.getSDeviation(variance)
        file.write(text + " variance: " + str(variance) + "\n")
        file.write(text + " standard deviation: " +
                   str(standardDeviation) + "\n")
        file.write(text + " average: " + str(average) + "\n\n")
        return

    def getVariance(self, population, average):
        """Function for getting the population variance of the initial generation

        Args:
            population (list[float]): A list of the initial generations fitness values
            average (float): The average fitness value of the population

        Returns:
            float: The population variance of the populations fitness
        """
        sum = 0
        for i in population:
            sum = sum + ((i - average) * (i - average))
        return sum / len(population)

    def getSDeviation(self, variance):
        """Function for getting the standard deviation of the initial generation

        Args:
            variance (float): The population variance of the populations fitness

        Returns:
            float: The standard deviation of the populations fitness
        """
        return math.sqrt(variance)

    def getAverage(self, population):
        """Function for getting the average fitness of the initial generation

        Args:
            population (list[float]): A list of the initial generations fitness values

        Returns:
            float: The average fitness value of the population
        """
        sum = 0
        for i in population:
            sum = sum + i
        return sum / len(population)
