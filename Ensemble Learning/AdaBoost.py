#Made by Devin Schwehr for CS 5350 Assignment 2
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

attribute_value_dictionary = {}
example_hits = []
example_misses = []
votes = []

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
    while (i < value_numbers.size):
        summation += (value_numbers[i]/s_size) * calculations[i]
        i += 1
    return total - summation 

def Get_Most_Common_Label(data):
    return data['label'].value_counts().idxmax()

def Get_Total_Value(label_values,num_rows):
    total_value = 0
    for value in label_values:
        total_value += Calc_Entropy(value, num_rows)
    return total_value

def Get_Root_Node_Entropy(data,attributes, total_entropy):
    best_attribute = None
    best_info_gain = None
    for attribute in attributes:
        if(attribute != 'label'):
            attribute_values = pd.unique(data[attribute])
            entropies = []
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
                value_entropy += Calc_Entropy(hit_data['weight'].sum(), filtered_data.shape[0])
                value_entropy += Calc_Entropy(miss_data['weight'].sum(), filtered_data.shape[0])
                entropies.append(value_entropy)
            attribute_info_gain = Info_Gain(total_entropy, data.shape[0], data[attribute].value_counts(), entropies)
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
        # hit_value = Calc_Entropy(hit_data['weight'].sum(),filtered_data.shape[0])
        # miss_value =Calc_Entropy(miss_data['weight'].sum(), filtered_data.shape[0])
        if hit_value > miss_value:
            root_node.next[value] = Node(label='yes')
        else:
            root_node.next[value] = Node(label = 'no')



# def Recursive_ID3(data, attributes, total_entropy, defined_depth):
#     if defined_depth == 0:
#         return Node(label = Get_Most_Common_Label(data))
#     #Part 1, checking if all labels are the same
#     if len(pd.unique(data['label'])) == 1:
#         return Node(label=pd.unique(data['label'])[0])
#     #Check if there are no more attributes to look on
#     if len(attributes) == 0:
#         return Node(label= Get_Most_Common_Label(data))
#     #This is for part 2 in the method
#     #Using a tuple we can get both the root node and its corresponding info gain, which we will call new_entropy
#     root_node, new_error = Get_Root_Node_Entropy(data, attributes, total_entropy)
#     for value in attribute_value_dictionary[root_node.attribute]:
#         #get all rows that contain that attribute value
#         is_val = data[root_node.attribute] == value
#         value_subset = data[is_val]
#         #Check to see if Sv is empty
#         length = len(value_subset.index)
#         if length == 0:
#             root_node.next[value] = Node(label= Get_Most_Common_Label(data))
#         else:
#             new_attributes = attributes[:]
#             new_attributes.remove(root_node.attribute)
#             new_depth = defined_depth -1
#             root_node.next[value] = Recursive_ID3(value_subset,new_attributes, new_error, new_depth)
#     return root_node

def Calculate_Stump(data,attributes,total_entropy):
    #First, get the current root node
    root_node, new_entropy = Get_Root_Node_Entropy(data,attributes,total_entropy)
    Get_Leaf_Node(data,root_node)
    # for value in attribute_value_dictionary[root_node.attribute]:
    #     root_node.next[value] = Get_Root_Node_Entropy(data,attributes,new_entropy)
    return root_node

def Adjust_Weight(data,vote):
    #Calculate the new constant to normalize the weight
    for hit in example_hits:
        #For that example, set it's new weight to the decremented old weight
        new_weight = data['weight'].iloc[hit] * np.exp(vote * -1)
        new_weight =  new_weight
        data['weight'].iloc[hit] = new_weight
        # data[hit,'weight'] = new_weight
    for miss in example_misses:
        #same as above, increase the weight instead though
        new_weight = data['weight'].iloc[miss] * np.exp(vote)
        new_weight =  new_weight
        data['weight'].iloc[miss] = new_weight
        # data[miss,'weight'] = new_weight
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
        #     example_misses.append(i)
        # example_hits.append(i)
        i += 1

    return wrong_predictions_weight,test_error

def Calculate_Final_Hypothesis(data,root_node):
    i = 0
    result = 0
    while i < data.shape[0]:
        current_node = root_node
        while current_node.label == None:
            current_node = current_node.next[data[current_node.attribute].iloc[i]]
        if current_node.label != data['label'].iloc[i]:
            result -= 1
        else:
            result += 1
    return result/data.shape[0]

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
    data = pd.read_csv(r"C:\Users\devin\OneDrive\Documents\CS5350\CS5350\Ensemble Learning\train.csv", header=None, names=bank_cols, delimiter=',')
    test_data = pd.read_csv(r"C:\Users\devin\OneDrive\Documents\CS5350\CS5350\Ensemble Learning\test.csv", header=None, names=bank_cols, delimiter=',')

    #Before we can begin the recursive function, we must eliminate numeric values from the Dataframe
    Remove_Numeric_Values(data, bank_cols)
    Remove_Numeric_Values(test_data, bank_cols)
    # Remove_Unknown_Values(data, bank_cols)
    # Remove_Unknown_Values(test_data, bank_cols)

    #we have to populate our global dictionary
    Populate_Global_Dictionary(data, bank_cols)

    #Here is where we will add the weights within the dataframe

    # Now that we have a Dataframe, calculate the total entropy
    num_rows = data.shape[0]
    total_label_values = data['label'].value_counts()
    total_error = Get_Total_Value(total_label_values, num_rows)
    #Now that we have our total entropy, we can begin our recursive method to find the tree.
    bank_cols.remove('label')

    data['weight'] = 1/num_rows
    # test_data['weight'] = 1/test_data.shape[0]
    # root_node = Recursive_ID3(data, bank_cols, total_error, tree_depth)
    i = 1
    error = total_error
    while i <= iterations:
        root_node = Calculate_Stump(data,bank_cols,error)
        #Now that we have the root node, we can calculate the accuracy of the tree
        train_error,test_error = Calculate_Accuracies(root_node, data,test_data)
        # test_error = Calculate_Accuracy(root_node, test_data)
        error = train_error
        print('after ' + str(i) +' :')
        print('training error = ' + str(train_error) + '  test error = ' + str(test_error))
        #Calculate it's vote via the training error:
        vote = (1/2)*np.log((1-train_error)/train_error)
        votes.append(vote)
        Adjust_Weight(data,vote)
        #Now, using the stump we calculate the final hypothesis
        final_hypothesis = Calculate_Final_Hypothesis(data,root_node)
        print('final hypothesis: ' + str(final_hypothesis))
        i+= 1


    print('program finished after ' + str(iterations) + ' iterations')
    # print('training error = ' + str(train_error) + '  test error = ' + str(test_error))

if __name__ == "__main__":
    main()
