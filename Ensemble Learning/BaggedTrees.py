from os import replace
import numpy as np
import pandas as pd
import ID3Numerical as dt

def main():
    #Goal is to use Pandas to generate the table.
    bank_cols = ['age','job','marital','education','default','balance','housing', 'loan', 'contact',
    'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']
    data = pd.read_csv(r"C:\Users\devin\OneDrive\Documents\CS5350\CS5350\DecisionTree\bank_files\train.csv", header=None, names=bank_cols, delimiter=',')
    test_data = pd.read_csv(r"C:\Users\devin\OneDrive\Documents\CS5350\CS5350\DecisionTree\bank_files\test.csv", header=None, names=bank_cols, delimiter=',')

    #Before we can begin the recursive function, we must eliminate numeric values from the Dataframe
    dt.Remove_Numeric_Values(data, bank_cols)
    dt.Remove_Numeric_Values(test_data, bank_cols)

    #we have to populate our global dictionary
    dt.Populate_Global_Dictionary(data, bank_cols)

    # Now that we have a Dataframe, calculate the total entropy
    num_rows = data.shape[0]
    total_label_values = data['label'].value_counts()
    total_error = dt.Get_Total_Value(total_label_values, num_rows, 1)
    #Now that we have our total entropy, we can begin our recursive method to find the tree.
    bank_cols.remove('label')
    train_error = 0
    test_error = 0
    iterations = 1
    while iterations <= 500:
        sample_data = data.sample(n=1000, replace=True)
        root_node = dt.Recursive_ID3(sample_data, bank_cols, total_error, 100, 1)
        train_error += dt.Calculate_Accuracy(root_node, data)
        test_error += dt.Calculate_Accuracy(root_node, test_data)
        print('at ' + str(iterations) + ' iterations: ')
        print('train error = ' + str(train_error/iterations) + '  test error: ' + str(test_error/iterations))
        iterations += 1

    print('program finished after 500 iterations')


if __name__ == "__main__":
    main()
