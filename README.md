# CS5350
This is a machine learning library developed Devin Schwehr for CS5350/6350 in University of Utah

There are two scripts present in the repository. run-binary.sh and run-numerical.sh

run-binary.sh runs the python program (ID3.py) that handles the dataset present in Problem 2: the car data set.

run-numerical.sh runs the python program (ID3-Numerical.py) that handles the dataset present in Problem 3: the bank data set. It is also built to automatically remove 
any numerical values and 'unknown' values present in the dataset.

At launch, both of them will ask you for the tree depth. Please input the number corresponding to the depth you want and press enter. After that,
It will ask you how you want to calculate the information gain. There are 3 valid inputs:
1 - Entropy (The log eqaution)
2 - Gini Index
3 - Majority Error
type in the number for which one you want to use and press enter.

After a few moments (it can take a few seconds on run-numerical.sh) the program will report that it has finished building the tree for the corresponding depth.
It will also give you the calculated training error and the calculated test error. 