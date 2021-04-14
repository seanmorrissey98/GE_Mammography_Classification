# Machine Learning using Grammatical Evolution Project
--------------
## Fitness Files
Four fitness files were created for use throughout this project which include:
- `SOO.py`
- `MultiObjective.py`
- `NoPIRS.py`
- `MonteCarlo.py`

All of the above fitness files were created for use in the different experiments of this project. Each of the four fitness files are suited for a specific use case which will be mentioned below.

#### **`SOO.py`**
--------------
As its name suggests, this fitness file was used in the experiment to discover which fitness function created the best classifiers in terms of **TPR** and **AUC** on the test set using _Single Objective Optimisation_. To change the fitness function used in this fitness file simply change the following line in the `evaluate()` function, for example below changes the fintess file to use the AUC fitness function.
```
fitness = self.getRocAucScore(progOuts)
```
or to use TPR as the fitness function change it to:
```
fitness = self.getTruePositiveRate(progOuts)
```
or to use MCC as the fitness function change it to:
```
fitness = self.getMCC(progOuts)
```
or to use AVGA as the fitness function change it to:
```
fitness = self.getAVGA(progOuts)
```

To use this fitness file make sure to change the following file and its parameter as seen below:
###### **`single.txt`**
```
FITNESS_FUNCTION:       SOO
```
Then you can run PonyGE2 from the **source directory** using the following commands:
```
cd src/
python ponyge.py --parameters single.txt
```
#### **`MultiObjective.py`**
--------------
This fitness file was used in two primary experiments. The first being the experiment to examine the effects of different sampling techniques on fitness and the second being the experiment to produce the best classifier as possible. This fitness file uses both the **OICB** boundary determination technique and the **PIRS** sampling technique.

To use this fitness file make sure to change the following file and its parameter as seen below:
###### **`multi.txt`**
```
FITNESS_FUNCTION:       MultiObjective
```
Then you can run PonyGE2 from the **source directory** using the following commands:
```
cd src/
python ponyge.py --parameters multi.txt
```
#### **`NoPIRS.py`**
--------------
This fitness file was also used in two primary experiments. The first being the experiment to examine the effects of different boundary determination techniques on fitness, average tree size and average nodes per tree and the second being the experiment to examine the effects of different sampling techniques on fitness. This fitness file has the potential to use either **OICB** for boundary determination or **Static boundaries** by changing only one line. This fitness file also has the potential to use either the **original dataset** or **oversampled dataset** by changing few lines.

To use the **OICB** algorithm for boundary determination simply ensure that the following line is uncommented in the `evaluate()` function as follows:
```
self.getBoundary(min, max, initMid, initMin, initMax, error, progOuts)
```
or to use **Static Boundaries** instead make sure to comment the following line:
```
#self.getBoundary(min, max, initMid, initMin, initMax, error, progOuts)
```

To use the **Original Dataset** for sampling simply ensure that the following line is in the `__init__()` function and the **training if block** in the `evaluate()` function as follows:
```
in_file = "../data/haralick02_50K.csv"
```
or to use **Oversampled Dataset** instead simply ensure that the following line is in the `__init__()` function and the **training if block** in the `evaluate()` function as follows:
```
in_file = "../data/haralick_preparedV2.csv"
```
To use this fitness file make sure to change the following file and its parameter as seen below:
###### **`multi.txt`**
```
FITNESS_FUNCTION:       NoPIRS
```
Then you can run PonyGE2 from the **source directory** using the following commands:
```
cd src/
python ponyge.py --parameters multi.txt
```
#### **`MonteCarlo.py`**
--------------

As its name suggests, this fitness file was developed for use in Monte Carlo Simulations. The difference between this fitness file and other _Multi-Objective Optimisation_ fitness files mentioned above is that it gathers statistics and metrics for the first initial generation of 10,000 such as **population variance**, **standard deviation**, and **average fitness** using all 4 of the fitness functions implemented in this project (TPR, AVGA, AUC, MCC).

To use this fitness file make sure to change the following file and its parameter as seen below:
###### **`multi.txt`**
```
FITNESS_FUNCTION:       MonteCarlo
POPULATION_SIZE:        10000
```
Then you can run PonyGE2 from the **source directory** using the following commands:
```
cd src/
python ponyge.py --parameters multi.txt
```