#Made by Devin Schwehr for CS 5350 Assignment 2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'

attribute_value_dictionary = {}
example_hits = []
example_misses = []
votes = []
root_nodes = []
train_hypotheses = []
test_hypotheses = []
ensemble_example_signs = {}

def Get_Num_Values(dataframe, value):
    return dataframe.loc[dataframe['label'] == value].count()

class Node:
    def __init__(self, attribute=None, values=None, label=None):
        self.attribute = attribute
        self.values = values
        self.next = {}
        #This is for leaf nodes
        self.label = label

#This method is to calculate the Entropy given the set of probabilities
def Calc_Entropy(values, size):
    return (values/size)*np.log2(values/size) * -1

def Info_Gain(total, s_size, value_numbers, calculations):
    summation = 0
    i = 0
    while (i < len(value_numbers)):
        summation += (value_numbers[i]/s_size) * calculations[i]
        i += 1
    return total - summation 

def Get_Most_Common_Label(data):
    return data['label'].value_counts().idxmax()

def Get_Total_Value(label_values,num_rows,data):
    total_value = 0
    pos_bool = data['label'] == 'yes'
    neg_bool = data['label'] == 'no'
    pos_data = data[pos_bool]
    neg_data = data[neg_bool]
    
    # for value in label_values:
    #     total_value += Calc_Entropy(value, num_rows)
    total_value += Calc_Entropy(label_values[1], num_rows) * pos_data['weight'].sum()
    total_value += Calc_Entropy(label_values[0],num_rows) * neg_data['weight'].sum()
    return total_value

def Get_Root_Node_Entropy(data,attributes, total_entropy):
    best_attribute = None
    best_info_gain = None
    for attribute in attributes:
        if(attribute != 'label'):
            attribute_values = pd.unique(data[attribute])
            entropies = []
            value_weights = []
            #Gather all of the entropies
            for value in attribute_values:
                val_bool = data[attribute] == value
                filtered_data = data[val_bool]
                yes_bool = filtered_data['label'] == 'yes'
                no_bool = filtered_data['label'] == 'no'
                #Separate the hit data and the miss data
                hit_data = filtered_data[yes_bool]
                miss_data = filtered_data[no_bool]
                value_entropy = 0
                #get the entropy for the yes/no data via summing the weight column
                hit_num = hit_data['weight'].sum()
                miss_num = miss_data['weight'].sum()
                value_weights.append(hit_num + miss_num)
                value_entropy += Calc_Entropy(hit_data['weight'].sum(), hit_data.shape[0])
                value_entropy += Calc_Entropy(miss_data['weight'].sum(), miss_data.shape[0])
                entropies.append(value_entropy)
            attribute_info_gain = Info_Gain(total_entropy, data.shape[0], value_weights, entropies)
            # attribute_info_gain = Info_Gain(total_entropy, data.shape[0], data[attribute].value_counts(), entropies)
            if best_info_gain == None or attribute_info_gain > best_info_gain:
                best_attribute = attribute
                best_info_gain = attribute_info_gain
    #Now that we have our best attribute, create our root node
    root_node = Node(best_attribute, pd.unique(data[best_attribute]))
    return root_node, best_info_gain

def Get_Leaf_Node(data,root_node):
    attribute = root_node.attribute
    attribute_values = pd.unique(data[attribute])
    for value in attribute_values:
        val_bool = data[attribute] == value
        filtered_data = data[val_bool]
        yes_bool = filtered_data['label'] == 'yes'
        no_bool = filtered_data['label'] == 'no'
        #Separate the hit data and the miss data
        hit_data = filtered_data[yes_bool]
        miss_data = filtered_data[no_bool]
        value_entropy = 0
        #get the entropy for the yes/no data via summing the weight column
        hit_value = hit_data['weight'].sum()
        miss_value = miss_data['weight'].sum()
        if hit_value > miss_value:
            root_node.next[value] = Node(label='yes')
        else:
            root_node.next[value] = Node(label = 'no')


def Calculate_Stump(data,attributes,total_entropy):
    #First, get the current root node
    root_node, new_entropy = Get_Root_Node_Entropy(data,attributes,total_entropy)
    Get_Leaf_Node(data,root_node)
    return root_node

def Adjust_Weight(data,vote):
    #Calculate the new constant to normalize the weight
    for hit in example_hits:
        #For that example, set it's new weight to the decremented old weight
        new_weight = data['weight'].iloc[hit] * np.exp(vote * -1)
        data['weight'].iloc[hit] = new_weight
    for miss in example_misses:
        #same as above, increase the weight instead though
        new_weight = data['weight'].iloc[miss] * np.exp(vote)
        data['weight'].iloc[miss] = new_weight
    equal_constant = data['weight'].sum()
    data['weight'] = data['weight']/equal_constant


def Calculate_Accuracies(root_node, data, test_data):
    wrong_predictions_weight = 0
    i = 0
    while i < data.shape[0]:
        current_node = root_node
        while current_node.label == None:
            current_node = current_node.next[data[current_node.attribute].iloc[i]]
        if current_node.label != data['label'].iloc[i]:
            wrong_predictions_weight += data['weight'].iloc[i]
            example_misses.append(i)
        else:
            example_hits.append(i)    
        i += 1
    i = 0
    test_error = 0
    while i < data.shape[0]:
        current_node = root_node
        while current_node.label == None:
            current_node = current_node.next[test_data[current_node.attribute].iloc[i]]
        if current_node.label != test_data['label'].iloc[i]:
            test_error += data['weight'].iloc[i]
        i += 1

    return wrong_predictions_weight,test_error

def Calculate_Final_Hypothesis(data,test_data):
    i = 0
    train_error = 0
    test_error = 0
    while i < data.shape[0]:
        training_result = 0
        j = 0
        while j < len(votes):
            current_node = root_nodes[j]
            while current_node.label == None:
                current_node = current_node.next[data[current_node.attribute].iloc[i]]
            if(current_node.label == 'yes'):
                training_result += votes[j]
            else:
                training_result -= votes[j]
            j += 1
        training_result = np.sign(training_result)
        ground_truth = data['label'].iloc[i]
        if (training_result > 0 and ground_truth == 'no') or (training_result < 0 and ground_truth == 'yes'):
            train_error += 1
        ground_truth = test_data['label'].iloc[i]
        if (training_result > 0 and ground_truth == 'no') or (training_result < 0 and ground_truth == 'yes'):
            test_error += 1
        i+= 1
    return train_error/data.shape[0], test_error/test_data.shape[0]

def Calc_Final_Hypothesis(data,test_data,root_node,vote):
    i = 0
    training_result = train_error = test_error = 0
    while i < data.shape[0]:
        training_result = 0
        current_node = root_node
        while current_node.label == None:
            current_node = current_node.next[data[current_node.attribute].iloc[i]]
        if(current_node.label == 'yes'):
            training_result += vote
        else:
            training_result -= vote
        ensemble_example_signs[i] +=training_result
        example_result = np.sign(ensemble_example_signs[i])
        ground_truth = data['label'].iloc[i]
        if (example_result > 0 and ground_truth == 'no') or (example_result < 0 and ground_truth == 'yes'):
            train_error += 1
        ground_truth = test_data['label'].iloc[i]
        if (example_result > 0 and ground_truth == 'no') or (example_result < 0 and ground_truth == 'yes'):
            test_error += 1
        i+= 1
    return train_error/data.shape[0], test_error/test_data.shape[0]


def Populate_Global_Dictionary(data, columns):
    for column in columns:
        attribute_value_dictionary[column] = pd.unique(data[column])


def Remove_Numeric_Values(data, column_names):
    for column in column_names:
        if data.dtypes[column] == np.int64:
            median_value = np.median(data[column].values)
            i = 0
            while i < len(data[column].values):
                if int(data[column].iloc[i]) < median_value:
                    data[column].iloc[i] = '-'
                else:
                    data[column].iloc[i] = '+'
                i += 1

def Remove_Unknown_Values(data, column_names):
    for column in column_names:
        if 'unknown' in pd.unique(data[column]):
            value_counts = data[column].value_counts()
            value_counts = value_counts.drop(labels=['unknown'])
            most_common_value = value_counts.idxmax()
            data[column].replace({'unknown': most_common_value}, inplace=True)


def main():
    print('Please Input The Number of Stumps')
    iterations = int(input())
    
    #Goal is to use Pandas to generate the table.
    bank_cols = ['age','job','marital','education','default','balance','housing', 'loan', 'contact',
    'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']
    data = pd.read_csv(r"./train.csv", header=None, names=bank_cols, delimiter=',')
    test_data = pd.read_csv(r"./test.csv", header=None, names=bank_cols, delimiter=',')

    #Before we can begin the recursive function, we must eliminate numeric values from the Dataframe
    Remove_Numeric_Values(data, bank_cols)
    Remove_Numeric_Values(test_data, bank_cols)

    #we have to populate our global dictionary
    Populate_Global_Dictionary(data, bank_cols)

    # Now that we have a Dataframe, calculate the total entropy
    num_rows = data.shape[0]
    total_label_values = data['label'].value_counts()
    #Now that we have our total entropy, we can begin our recursive method to find the tree.
    bank_cols.remove('label')

    data['weight'] = 1/num_rows
    i = 1
    part2a = open("part2a.txt", "a")
    train_errors = []
    test_errors = []
    final_trains=  []
    final_tests=  []
    iter = []
    j = 0
    while j < data.shape[0]:
        ensemble_example_signs[j] = 0
        j+=1
    while i <= iterations:
        total_error = Get_Total_Value(total_label_values, num_rows,data)
        error = total_error
        iter.append(i)
        root_node = Calculate_Stump(data,bank_cols,error)
        root_nodes.append(root_node)
        #Now that we have the root node, we can calculate the accuracy of the tree
        train_error,test_error = Calculate_Accuracies(root_node, data,test_data)
        train_errors.append(train_error)
        test_errors.append(test_error)
        error = train_error
        print('after ' + str(i) +' :')
        part2a.write('after ' + str(i) +' :\n')
        print('training error = ' + str(train_error) + '  test error = ' + str(test_error))
        part2a.write('training error = ' + str(train_error) + '  test error = ' + str(test_error) + '\n')
        #Calculate it's vote via the training error:
        vote = (1/2)*np.log((1-train_error)/train_error)
        votes.append(vote)
        final_train,final_test = Calc_Final_Hypothesis(data,test_data,root_node,vote)
        final_trains.append(final_train)
        final_tests.append(final_test)
        Adjust_Weight(data,vote)
        #Now, using the stump we calculate the final hypothesis
        print('final training hypothesis: ' + str(final_train) + '  final test hypothesis: ' + str(final_test))
        part2a.write('final training hypothesis: ' + str(final_train) + '  final test hypothesis: ' + str(final_test) + '\n')
        i+= 1


    print('program finished after ' + str(iterations) + ' iterations')
    part2a.write('program finished after ' + str(iterations) + ' iterations\n')
    part2a.close()
    print('training error = ' + str(train_error) + '  test error = ' + str(test_error))
    plt.figure(0)
    plt.plot(iter,train_errors, label = "Train Errors")
    plt.plot(iter,test_errors, label = "Test Errors")
    plt.xlabel('Iterations')
    plt.ylabel('Errors')
    plt.title('Individual Training and Test Errors')
    plt.legend()
    plt.show()
    plt.figure(1)
    plt.plot(iter,final_trains, label = "Ensemble Train Errors")
    plt.plot(iter,final_tests, label = "Ensemble Test Errors")
    plt.xlabel('Iterations')
    plt.ylabel('Errors')
    plt.title('Ensemble Training and Test Errors')
    plt.legend()
    plt.show()
if __name__ == "__main__":
    main()
