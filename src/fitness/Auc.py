from fitness.base_ff_classes.base_ff import base_ff
import pandas as pd
from sklearn.metrics import roc_auc_score

class Auc(base_ff):
    """
    derived from Py-max
    """
    maximise = True  # True as it ever was.

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()

        in_file = "C:/Users/seanm/Desktop/GE_Mammography_Classification/data/haralick02_50K.csv"
        df = pd.read_csv(in_file)
        haralick_features = []
        for i in range(104):
            feature = "x"+ str(i)
            haralick_features.append(feature)

        self.data = df[haralick_features]

        training_size = round(len(self.data) * .20)
        self.training = self.data[:-training_size]
        self.test = self.data
        self.n_vars = len(self.data)

        self.training_test = True
    
    def evaluate(self, ind, **kwargs):
        # ind.phenotype will be a string, including function definitions etc.
        # When we exec it, it will create a value XXX_output_XXX, but we exec
        # inside an empty dict for safety.

        dist = kwargs.get('dist', 'training')
        data = []
        start = 0

        if dist == "training":
            # Set training datasets.
            data = self.training
            start = round(4999 * .20)

        elif dist == "test":
            # Set test datasets.
            data = self.test
            start = len(self.test) - round(4999 * .20)

        p, d = ind.phenotype, {}
        n_points = len(data)  # Number of data points available . . 4999
        in_file = "C:/Users/seanm/Desktop/GE_Mammography_Classification/data/haralick02_50K.csv"
        df = pd.read_csv(in_file)
        labels = df['Label']
        predictions = []
        for i in range(start, n_points):
            main = []
            opposite = []
            for j in range(52):
                main.append(data["x"+str(j)][i])
                opposite.append(data["x"+str(j+52)][i])
            d["main"] = main
            d["opposite"] = opposite
            d['n_points'] = len(d['main'])

            exec(p, d)
            if d["XXX_output_XXX"] > 0:  # Guessing suspicious area present
                predictions.append(1)
            else:  # Guessing suspicious area not present
                predictions.append(0)

        return roc_auc_score(labels[start:n_points], predictions)