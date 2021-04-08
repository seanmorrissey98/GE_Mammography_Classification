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

## Features

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
## Results
