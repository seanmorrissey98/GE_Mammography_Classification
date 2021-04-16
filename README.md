# Machine Learning using Grammatical Evolution Project
--------------
## Abstract

An approach to automating the detection of suspicious areas in mammograms
through the development of a stage 1 computer aided detector using GE as its
fundament to develop classifiers is described.

Mammography is a process which uses X-rays to identify whether breast cancer is
present in patients. Mammograms are evaluated by doctors and radiologists but
with the use of digital mammography becoming customary, it opens up for the
automation of this process to help in identifying and classifying regions of interests
and the presence of breast cancer.

A stage 1 computer aided detector can develop classifiers for mammograms by
feeding segments of mammograms into a GE workflow. The classifiers take textural
symmetry between both breasts into account when determining the presence of a
suspicious area. Early detection of breast cancer using mammograms can greatly
increase the effectiveness of treatment and survival rates in patients, making this
process vital to identify all potential cancerous regions and let none go undetected.

The classifiers developed by the GE process aim to have the highest possible
accuracy which is achievable in terms of a true positive rate of identifying suspicious
areas and on the contrary a reduced rate of false positives per segment of a
mammogram.

## How-to
To run the program, first change directory to the source directory.
```
cd src/
```
From the source directory the *PonyGE2* program can be ran with the following command where `filename` is replaced with the relevant parameter filename which you want to use.
```
python ponyge.py --parameters filename.txt
```
Statistics such as the `generation`, `average tree size`, etc., along with either the average fitness when using a *Single Objective Optimisation* parameter file or a list of the pareto fronts when using a *Multi-Objective Optimisation* parameter file are printed to the terminal at the end of each generation.

The results of each run are updated at each generation and can be found in the _GE_Mammography_Classification/results_ folder.

## Experiments
All experiments were ran using **Python 3.8.9**. For more information on a **how-to** and experiments please read the [**README.md**](src/fitness/README.md) in the _GE_Mammography_Classification/src/fitness_ folder.

## Features
#### Grammar
The grammars for this project can be found in the _GE_Mammography_Classification/grammars_ folder, along with more information. The primary grammar file used throughout this project was *FypV2.pybnf*. The non-terminal set for the grammar is what will build out the derivation trees of individuals as it contains recursive elements. It does this through a recursive amount of if statements. The if statements can select and combine Haralick features and using logical operators compare them against other Haralick features.

#### Parameters
The parameter files for this project can be found in the _GE_Mammography_Classification/parameters_ folder, along with more information. The primary parameter files used throughout this project were *single.txt* and *multi.txt*. The difference between these two parameter files are that *single.txt* is for use in *Single Objective Optimisation* and uses **generational** for _replacement_ and **tournament** for _selection_. On the other hand, *multi.txt* is for use in *Multi-Objective Optimisation* and uses **NSGA 2 replacement** and **NSGA 2 selection**.

#### Fitness Functions
Four fitness functions were implemented as part of this project:
* True Positive Rate (TPR)
* Area under the curve of receiver operating characteristics (AUC)
* Matthews Correlation Coefficient (MCC)
* Average Accuracy (AVG)

The fitness functions for this project can be found in the _GE_Mammography_Classification/src/fitness_ folder, along with more information.
###### True Positive Rate
The first fitness function taken into consideration was the true positive rate also known as Sensitivity i.e., the number of correctly identified suspicious areas per segment. The equation for the TPR can be seen in the image below.

![True Positive Rate](/images/TPR.png)

###### Area Under the Curve
The second fitness function taken into consideration needed to combine aspects of both the true positive rate and false positive rate and so the AUC of ROC curve was considered. This method is a “single scalar value that measures the overall performance of a binary classifier”. The true positive rate is plotted on the y axis, with the false positive rate plotted on the x axis. An example of a plot of an AUC can be seen in the image below.

![Area Under the Curve](/images/AUC.png)

###### Matthews Correlation Coefficient
The third fitness function taken into consideration was the Matthews Correlation Coefficient which is a common fitness function in machine learning for the use in binary classification problems. This fitness function calculates correlation coefficient between the actual and the predicted classifications. Unlike the TPR, the MCC takes into account true positives, true negatives, false positives and false negatives.The equation for the MCC can be seen in the image below.

![Matthews Correlation Coefficient](/images/MCC.png)

###### Average Accuracy
The final fitness function taken into consideration was Average Accuracy (AVGA) also known as Balanced Accuracy. The equation for AVGA works by adding the TPR (sensitivity) and the specificity (true negative rate) together and dividing it by 2 or multiplying it by 0.5 to get a value in the range [0, 1]. The equation for the AVGA can be seen in the image below.

![Average Accuracy](/images/AVGA.png)
#### Sampling
Two sampling techniques were implemented as part of this project to reduce the class imbalace problem:
* Oversampling
* Proportional Individualised Random Sampling (PIRS)

###### Oversampling
Oversampling aims to reduce the class imbalance by increasing the number of individuals in the minority class. The oversampling technique implemented for this project is random oversampling. Random oversampling works to increase the minority class by duplicating individuals in the minority class. A breakdown of the dataset before and after oversampling can be seen in the image below.

![Oversampling](/images/sampling.png)
###### Proportional Individualised Random Sampling
PIRS differs from oversampling as the size of the dataset remains the same as the original dataset and so does not increase the training time and computational cost / requirements which over sampling introduces due to a larger dataset. PIRS works by varying the number of instances of each class a classifier is trained with from the original dataset. This way each classifier in the population is trained on varying ratios of the minority to majority class.

#### Boundary Determination
Two boundary determination techniques were implemeted as part of this project to reduce bias and optimise results:
* Static Boundaries
* Optimised Individual Class Boundaries (OICB)

###### Static Boundaries
Static boundaries are continuous boundaries which stay the same for all individuals throughout a run in GP. Typically, static boundaries are set to zero (zero threshold). However, there are potential pitfalls to static boundaries. Firstly, they can potentially introduce bias as the determination of a good static boundary may require expertise. Also, “individuals will take some time to move in the direction of the zero boundary” when using a static boundary. An example of a static boundary showing potential bias can be seen in the image below.

![Static Boundaries](/images/static.png)
###### Optimised Individual Class Boundaries
OICB’s goal is to find the most optimal boundary for each individual. Due to OICB determining boundaries for each individual, it reduces the bias introduced from static boundaries as no matter how high or low program output is for a classifier, OICB works to find the optimal boundary. OICB calculates the optimal boundary by ordering the program output of an individual and recursively searching smaller ranges of the output while testing different boundary values. The error metric which OICB uses to decide whether one boundary is better than another is Classification Error.

![OICB](/images/OICB_BACKGROUND.png)
#### Monte Carlo Simulation
Monte Carlo simulation is a simulation technique which uses repeated random sampling, statistical analysis and modelling and can be used to mimic operations of a complex systems or mathematical operations. The use of a Monte Carlo simulation in this paper is to validate whether the fitness functions mentioned above were behaving correctly. This is done by generating a large number of individuals for the initial generation’s population and gathering statistical results on the populations fitness. A Monte Carlo Simulation is completed on each of the four fitness functions, MCC, AVGA, TPR and AUC in parallel, with each individual generated for the initial population being evaluated on all four of the fitness functions. The statistics gathered during Monte Carlo Simulations can be seen in the images below.

![Population Variance](/images/VARIANCE.png)

![Standard Deviation](/images/DEVIATION.png)

![Population Average](/images/AVERAGE.png)

## Results
A subset of the results achieved throughout this projectcan be found in the _GE_Mammography_Classification/results_ folder. Please note that more results can be made available through request.
