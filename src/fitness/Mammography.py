from fitness.base_ff_classes.base_ff import base_ff
import pandas as pd
import random

class Mammography(base_ff):
    """
    derived from Py-max
    """
    maximise = True  # True as it ever was.

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()

        in_file = "C:/Users/seanm/Desktop/PonyGE2/data/haralick02_50K.csv"
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
        in_file = "C:/Users/seanm/Desktop/PonyGE2/data/haralick02_50K.csv"
        df = pd.read_csv(in_file)
        labels = df['Label']
        tp, fn = 0,0
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
                if labels[i] == 1:
                    # Correct guess increase true positive counter
                    tp = tp + 1
            else:  # Guessing suspicious area not present
                if labels[i] == 1:
                    # Incorrect guess increase false negative counter
                    fn = fn + 1

        # Prevent division by 0
        fn = 1 if tp + fn == 0 else fn
        #print("Done. ", random.randint(0,1000))
        return tp/(tp+fn)


# --debug --grammar_file test.pybnf --verbose --fitness_function trading2 --generations 20 --population 20 --random_seed 517470
# add cache
# --target_seed_folder my_seeds