from os import replace
from random import triangular
import numpy as np
import pandas as pd
from pandas.core.tools.timedeltas import to_timedelta
import ID3Numerical as dt
import ID3NumericalForest as dtf

bagged_dict = {}
bagged_dict.setdefault(int, [])

def Gather_Bags(data, bank_cols, total_error):
    iter = 0
    bagged_dict[101] = []
    while iter < 5:
        subdata = data.sample(1000)
        tree_index = 0
        tree_list = []
        print('making trees')
        while tree_index < 5:
            root_node = dt.Recursive_ID3(subdata,bank_cols,total_error,100,1)
            tree_list.append(root_node)
            tree_index +=1
        print('made trees')
        bagged_dict[iter] = tree_list
        bagged_dict[101].append(bagged_dict[iter][0])
        print('appended 0th tree to single list')
        iter +=1

def Compute_Information(index,data):
    #First, we have to compute the bias 
    iter = 0
    i = 0
    total = []
    current_bias = 0
    current_variance = 0
    total_bias = 0
    total_variance = 0
    while i < data.shape[0]:
        iter = 0
        total = []
        while iter < 5:
            current_node = bagged_dict[index][iter]
            while current_node.label == None:
                current_node = current_node.next[data[current_node.attribute].iloc[i]]
            if(current_node.label == 'yes'):
                total.append(1)
            else:
                total.append(-1)
            iter +=1
        current_bias = np.sum(total)/100
        if(data['label'].iloc[i] == 'yes'):
            current_bias -= 1
        else:
            current_bias -= -1
        current_bias = np.square(current_bias)
        current_variance = Calculate_Variance(current_bias,total)
        total_variance += current_variance
        total_bias += current_bias
        i += 1
    total_bias /= data.shape[0]
    total_variance /=data.shape[0]
    return total_bias, total_variance

def Calculate_Variance(mean,predictions):
    iter = 0
    variance = 0
    while iter < 5:
        variance += np.square(predictions[0] - mean)
        iter +=1
    return variance/99

def Construct_Forest(data,test_data, bank_cols,total_error, split_size,text_file):
    i = 1
    train_error = 0
    test_error = 0
    while i <= 100:
        sample_data = data.sample(n=5000, replace=True)
        root_node = dtf.Recursive_ID3(sample_data,bank_cols,total_error,150,1,split_size)
        train_error += dtf.Calculate_Accuracy(root_node,data)
        test_error += dtf.Calculate_Accuracy(root_node,test_data)
        print('at ' + str(i) + ' iterations: ')
        print('train error = ' + str(train_error/i) + '  test error: ' + str(test_error/i))
        text_file.write('at ' + str(i) + ' iterations: \n')
        text_file.write('train error = ' + str(train_error/i) + '  test error: ' + str(test_error/i) + '\n')
        i+=1

def main():
    #Goal is to use Pandas to generate the table.
    bank_cols = ['age','job','marital','education','default','balance','housing', 'loan', 'contact',
    'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']
    data = pd.read_csv(r"C:\Users\devin\OneDrive\Documents\CS5350\CS5350\DecisionTree\bank_files\train.csv", header=None, names=bank_cols, delimiter=',')
    test_data = pd.read_csv(r"C:\Users\devin\OneDrive\Documents\CS5350\CS5350\DecisionTree\bank_files\test.csv", header=None, names=bank_cols, delimiter=',')

    #Before we can begin the recursive function, we must eliminate numeric values from the Dataframe
    dt.Remove_Numeric_Values(data, bank_cols)
    dt.Remove_Numeric_Values(test_data, bank_cols)

    dtf.Remove_Numeric_Values(data,bank_cols)
    dtf.Remove_Numeric_Values(test_data,bank_cols)

    #we have to populate our global dictionary
    dt.Populate_Global_Dictionary(data, bank_cols)
    dtf.Populate_Global_Dictionary(data, bank_cols)

    # Now that we have a Dataframe, calculate the total entropy
    num_rows = data.shape[0]
    total_label_values = data['label'].value_counts()
    total_error = dt.Get_Total_Value(total_label_values, num_rows, 1)
    #Now that we have our total entropy, we can begin our recursive method to find the tree.
    bank_cols.remove('label')
    train_error = 0
    test_error = 0
    iterations = 1
    # while iterations <= 500:
    #     sample_data = data.sample(n=5000, replace=True)
    #     root_node = dt.Recursive_ID3(sample_data, bank_cols, total_error, 200, 1)
    #     train_error += dt.Calculate_Accuracy(root_node, data)
    #     test_error += dt.Calculate_Accuracy(root_node, test_data)
    #     print('at ' + str(iterations) + ' iterations: ')
    #     print('train error = ' + str(train_error/iterations) + '  test error: ' + str(test_error/iterations))
    #     iterations += 1
    # Gather_Bags(data,bank_cols,total_error)
    # single_bias, single_variance = Compute_Information(101,data)
    # single_squared = single_bias + single_variance
    # print('Single Tree group bias: ' + str(single_bias) + ' variance: ' + str(single_variance))
    # print('Single tree group squared ' + str(single_squared))
    # i = 0
    # while i < 5:
    #     bagged_bias, bagged_variance = Compute_Information(i,data)
    #     bagged_squared = bagged_bias + bagged_variance
    #     print('Bagged Tree group bias: ' + str(bagged_bias) + ' variance: ' + str(bagged_variance))
    #     print('Bagged tree group squared ' + str(bagged_squared))
    #     i+=1
    
    #Now to utilize the Random Forests Method
    part2d = open("part2d.txt", "a")
    print('for attribute split size of 2')
    Construct_Forest(data,test_data, bank_cols, total_error, 2,part2d)
    print('for attribute split size of 4')
    Construct_Forest(data,test_data, bank_cols, total_error, 4,part2d)
    print('for attribute split size of 6')
    Construct_Forest(data,test_data, bank_cols, total_error, 6,part2d)
    part2d.close()
if __name__ == "__main__":
    main()
