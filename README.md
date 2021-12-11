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

FOR ASSIGNMENT 2:

There are 2 additional folders: Ensemble Learning and Linear Regression.

In Ensemble Learning are 2 sh files:
    - run-ada.sh will run the AdaBoost File
    - run-bagged.sh will run the BaggedTrees File

    For AdaBoost:
        Specify how many stumps you would like to generate and the program will then output each stump's individual errors as well as the current ensemble's 
        train and test error.

    For BaggedTrees:
        There are 4 different Modes you can run the file in:
            - By inputting 'Bagged Trees' it will create 500 trees and print the average error at each iteration.
            - By inputting 'Bagged Trees BV' it will perform the process in Problem 2c, namely gathering 100 bags of 100 trees and then computing the bias, variance,
            and group squared of the single tree's group (the 0th tree from each of the bags) as well as the bias,variance, and group square of each of the bags.
            - By inputting 'Forest' you will be prompted to input the Attribute Sample Size you want to split the attributes upon. It will then construct 100
            trees.
            - By inputting 'Forest BV' you will be prompted to input your Attribute Sample size, at which point it will follow the same process as it does for Bagged Trees BV.

In Linear Regression there is 1 sh file for the Linear-Regression File:
    - By Inputting 'Batch' it will perform Batch Descent
    - By Inputting 'Stochastic' it will perform Stochastic Descent
    It will always output the Analytical optimal weight vector
    
FOR ASSIGNMENT 3:

Perceptron has one sh file, titled 'run-perceptron.sh'

Running it will ask you to input a number that corresponds to each of the respective Perceptron types. Enter in the appropriate number and hit enter to get the desired output from the program.

FOR ASSIGNMENT 4:

SVM just has one sh file, titled 'svm.sh'

Running it asks for you to put in an input that corresponds to each of the different implementations. Put in the proper number and hit enter to begin execution.

There is the need for manual changing to get certain outcomes.
    - For Primal, in main you can pick which yt value you want by uncommenting the desired function and commenting out the other.
    - For Gaussian Dual, you need to remove the other C values and decrease the number of iterations to see the # of overlapping support vectors.

FOR ASSIGNMENT 5:

There are two sh files, 'NN.sh' and 'NN_PyTorch.sh'

Running NN.sh will ask you for a desired width. Inputting that will output the training and test error for that width. 

Running NN_PyTorch.sh will run the PyTorch Implementation for all widths. Both versions of activations are initialized to a depth of 3. To add more depths, you will need to copy and paste lines 16 & 17 for ReLu and lines 40 & 41 for Tanh to get the desired depth. Lines 104 & 105 dictate which activation you want to use. 

There is also a file, Prob4.py, that shows the logic for how I calculated the gradients in Problem 4 Part b. 
