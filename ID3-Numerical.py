#Made by Devin Schwehr for CS 5350 Assignment 1
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

attribute_value_dictionary = {}

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

#This method is used to calculate the Gini Index
def Calc_Gini(probability, size):
    return np.power(probability/size,2)

def Calc_Majority(probability, size):
    return (size-probability)/size

def Info_Gain(total, s_size, value_numbers, calculations):
    summation = 0
    i = 0
    while (i < value_numbers.size):
        summation += (value_numbers[i]/s_size) * calculations[i]
        i += 1
    return total - summation 

def Get_Most_Common_Label(data):
    return data['label'].value_counts().idxmax()

def Get_Total_Value(label_values,num_rows,decider):
    if decider ==1:
        total_value = 0
        for value in label_values:
            total_value += Calc_Entropy(value, num_rows)
        return total_value
    if decider ==2:
        total_value = 1
        for value in label_values:
            total_value -= Calc_Gini(value, num_rows)
        return total_value
    if decider == 3:
        total_value = 0
        for value in label_values:
            total_value += (num_rows-value)/num_rows
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
                label_counts = filtered_data['label'].value_counts()
                value_entropy = 0
                for label in label_counts:
                    value_entropy += Calc_Entropy(label, filtered_data.shape[0])
                entropies.append(value_entropy)
            attribute_info_gain = Info_Gain(total_entropy, data.shape[0], data[attribute].value_counts(), entropies)
            if best_info_gain == None or attribute_info_gain > best_info_gain:
                best_attribute = attribute
                best_info_gain = attribute_info_gain
    #Now that we have our best attribute, create our root node
    root_node = Node(best_attribute, pd.unique(data[best_attribute]))
    return root_node, best_info_gain

def Get_Root_Node_Gini(data,attributes, total_gini):
    best_attribute = None
    best_info_gain = None
    for attribute in attributes:
        if(attribute != 'label'):
            attribute_values = pd.unique(data[attribute])
            gini_indexes = []
            #Gather all of the entropies
            for value in attribute_values:
                val_bool = data[attribute] == value
                filtered_data = data[val_bool]
                label_counts = filtered_data['label'].value_counts()
                value_gini = 1
                for label in label_counts:
                    value_gini -= Calc_Gini(label, filtered_data.shape[0])
                gini_indexes.append(value_gini)
            attribute_info_gain = Info_Gain(total_gini, data.shape[0], data[attribute].value_counts(), gini_indexes)
            if best_info_gain == None or attribute_info_gain > best_info_gain:
                best_attribute = attribute
                best_info_gain = attribute_info_gain
    #Now that we have our best attribute, create our root node
    root_node = Node(best_attribute, pd.unique(data[best_attribute]))
    return root_node, best_info_gain

def Get_Root_Node_Majority(data,attributes, total_majority):
    best_attribute = None
    best_info_gain = None
    for attribute in attributes:
        if(attribute != 'label'):
            attribute_values = pd.unique(data[attribute])
            majority_errors = []
            #Gather all of the entropies
            for value in attribute_values:
                val_bool = data[attribute] == value
                filtered_data = data[val_bool]
                label_counts = filtered_data['label'].value_counts()
                value_majority = 1
                for label in label_counts:
                    value_majority += Calc_Majority(label, filtered_data.shape[0])
                majority_errors.append(value_majority)
            attribute_info_gain = Info_Gain(total_majority, data.shape[0], data[attribute].value_counts(), majority_errors)
            if best_info_gain == None or attribute_info_gain > best_info_gain:
                best_attribute = attribute
                best_info_gain = attribute_info_gain
    #Now that we have our best attribute, create our root node
    root_node = Node(best_attribute, pd.unique(data[best_attribute]))
    return root_node, best_info_gain



def Recursive_ID3(data, attributes, total_entropy, defined_depth, decider):
    if defined_depth == 0:
        return Node(label = Get_Most_Common_Label(data))
    #Part 1, checking if all labels are the same
    if len(pd.unique(data['label'])) == 1:
        return Node(label=pd.unique(data['label'])[0])
    #Check if there are no more attributes to look on
    if len(attributes) == 0:
        return Node(label= Get_Most_Common_Label(data))
    #This is for part 2 in the method
    #Using a tuple we can get both the root node and its corresponding info gain, which we will call new_entropy
    if decider == 1:
        root_node, new_error = Get_Root_Node_Entropy(data, attributes, total_entropy)
    if decider == 2:
        root_node, new_error = Get_Root_Node_Gini(data, attributes, total_entropy)
    if decider == 3:
        root_node, new_error = Get_Root_Node_Majority(data, attributes, total_entropy)
    for value in attribute_value_dictionary[root_node.attribute]:
        #get all rows that contain that attribute value
        is_val = data[root_node.attribute] == value
        value_subset = data[is_val]
        #Check to see if Sv is empty
        length = len(value_subset.index)
        if length == 0:
            root_node.next[value] = Node(label= Get_Most_Common_Label(data))
            # return Node(label= Get_Most_Common_Label(data))
        else:
            new_attributes = attributes[:]
            new_attributes.remove(root_node.attribute)
            new_depth = defined_depth -1
            # root_node.next.append(Recursive_ID3(value_subset, new_attributes , new_error, new_depth, decider))
            root_node.next[value] = Recursive_ID3(value_subset,new_attributes, new_error, new_depth, decider)
    return root_node

def Calculate_Accuracy(root_node, data):
    wrong_predictions = 0
    i = 0
    while i < data.shape[0]:
        current_node = root_node
        while current_node.label == None:
            current_node = current_node.next[data[current_node.attribute].iloc[i]]
        if current_node.label != data['label'].iloc[i]:
            wrong_predictions += 1
        i += 1
    return wrong_predictions/data.shape[0]

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
    print('Please Input the Tree Depth')
    tree_depth = int(input())
    print('Please Input how you want to select the attribute. 1=Entropy, 2=Gini, 3=Majority')
    decider = int(input())

    #Goal is to use Pandas to generate the table.
    bank_cols = ['age','job','marital','education','default','balance','housing', 'loan', 'contact',
    'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']
    data = pd.read_csv(r"./DecisionTree/bank_files/train.csv", header=None, names=bank_cols, delimiter=',')
    test_data = pd.read_csv(r"./DecisionTree/bank_files/test.csv", header=None, names=bank_cols, delimiter=',')

    #Before we can begin the recursive function, we must eliminate numeric values from the Dataframe
    Remove_Numeric_Values(data, bank_cols)
    Remove_Numeric_Values(test_data, bank_cols)
    Remove_Unknown_Values(data, bank_cols)
    Remove_Unknown_Values(test_data, bank_cols)

    #we have to populate our global dictionary
    Populate_Global_Dictionary(data, bank_cols)

    # Now that we have a Dataframe, calculate the total entropy
    num_rows = data.shape[0]
    total_label_values = data['label'].value_counts()
    total_error = Get_Total_Value(total_label_values, num_rows, decider)
    #Now that we have our total entropy, we can begin our recursive method to find the tree.
    bank_cols.remove('label')

    root_node = Recursive_ID3(data, bank_cols, total_error, tree_depth, decider)

    #Now that we have the root node, we can calculate the accuracy of the tree
    train_error = Calculate_Accuracy(root_node, data)
    test_error = Calculate_Accuracy(root_node, test_data)
    print('program finished with depth of ' + str(tree_depth))
    print('training error = ' + str(train_error) + '  test error = ' + str(test_error))

if __name__ == "__main__":
    main()


