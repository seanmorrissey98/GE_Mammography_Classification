# Machine Learning using Grammatical Evolution Project
--------------
## Parameters
Two parameter files were created for use throughout this project which include:
- `single.txt`
- `multi.txt`

The key differences between these two parameter files is that `single.txt` was built and created for _Single Objective Optimisation_, while  `multi.txt` was built and created for _Multi-Objective Optimisation_.

#### **`single.txt`**
As mentioned above this parameter file was built and created for _Single Objective Optimisation_ i.e., when using a single fitness function at a time. The key differences between this parameter file and the other is the _Replacement_ and _Selection_ algorithms, which can be seen below:
```
REPLACEMENT:            generational
SELECTION:              tournament
```
These two parameters cannot be used for _Multi-Objective Optimisation_ and was the reason for the creation of a separate parameter file.
#### **`multi.txt`**
As mentioned above this parameter file was built and created for _Multi-Objective Optimisation_ i.e., when using multiple fitness function at a time. The key differences between this parameter file and the other is the _Replacement_ and _Selection_ algorithms, which can be seen below:
```
REPLACEMENT:            nsga2_replacement
SELECTION:              nsga2_selection
```
As mentioned above the _generational_ and _tournament_ algorithms are not suitable for **MOO** and so were replaced by _NSGA2 Replacement_ and _NSGA2 Selection_. These are suitable for **MOO** and allow for the use of **Pareto Fronts**. 