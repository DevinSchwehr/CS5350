#Made by Devin Schwehr for CS 5350 Assignment 1
import numpy as np
import pandas as pd
import os

def Get_Num_Values(dataframe, value):
    return dataframe.loc[dataframe['label'] == value].count()

class Node:
    def __init__(self, attribute=None, values=None, label=None):
        self.attribute = attribute
        self.values = values
        self.next = None
        #This is for leaf nodes
        self.label = label

#This method is to calculate the Entropy given the set of probabilities
def Calc_Entropy(values, size):
    # result = 0
    # for prob in values:
    #     result += np.log(prob/size) * -1
    # return result
    return (values/size)*np.log(values/size) * -1

#This method is used to calculate the Gini Index
def Calc_Gini(probabilities, size):
    result = 1
    for prob in probabilities:
        result -= np.exp(prob/size)
    return result

def Info_Gain(total, s_size, value_numbers, calculations):
    summation = 0
    i = 0
    while (i < value_numbers.size):
        summation += (value_numbers[i]/s_size) * calculations[i]
        i += 1
    return total - summation 

def Get_Most_Common_Label(data):
    return data['label'].value_counts().index

def Get_Root_Node(data, total_entropy):
    best_attribute = None
    best_info_gain = None
    for attribute in data:
        attribute_values = data[attribute].value_counts()
        entropies = []
        #Gather all of the entropies
        for value in attribute_values:
            entropies.append(Calc_Entropy(value, data.shape[0]))
        attribute_info_gain = Info_Gain(total_entropy, data.shape[0], attribute_values, entropies)
        if attribute_info_gain > best_info_gain or best_info_gain == None:
            best_attribute = attribute
    #Now that we have our best attribute, create our root node
    root_node = Node(best_attribute, data[attribute].tolist())
    return root_node, best_info_gain

def Recursive_ID3(data, total_entropy):
    #Part 1, checking if all labels are the same
    if len(data.unique(data['label'])) == 1:
        return Node(label=data.unique(data['label'])[0])
    #Check if there are no more attributes to look on
    if data.shape[1] == 1:
        return Node(label= Get_Most_Common_Label(data))
    #This is for part 2 in the method
    #Using a tuple we can get both the root node and its corresponding info gain, which we will call new_entropy
    root_node, new_entropy = Get_Root_Node(data,total_entropy)
    for value in root_node.values:
        #First get all rows that contain that attribute value
        value_subset = data[data[root_node.attribute] == value]
        #Remove the Attribute from the list to get Sv
        value_subset.drop(root_node.attribute)
        #Check to see if Sv is empty
        if len(value_subset.index) == 0:
            leaf_node = Node(label= Get_Most_Common_Label(data))
            root_node.next = leaf_node
        root_node.next = Recursive_ID3(value_subset,new_entropy)
    return root_node

def main():
    print('Welcome to the program')
    #Goal is to use Pandas to generate the table.
    car_cols = ['buying','maint','doors','persons','lug_boot','safety','label']
    data = pd.read_csv(r"DecisionTree\train.csv", header=None, names=car_cols, delimiter=',')
    # Now that we have a Dataframe, calculate the total entropy
    num_rows = data.shape[0]
    # total_label_values = [Get_Num_Values(data,'unacc'), Get_Num_Values(data,'acc'), Get_Num_Values(data,'good'),
    # Get_Num_Values(data,'vgood')]
    total_label_values = data['label'].value_counts()
    total_entropy = 0
    for value in total_label_values:
       total_entropy += Calc_Entropy(value, num_rows)
    #Now that we have our total entropy, we can begin our recursive method to find the tree.
    root_node = Recursive_ID3(data, total_entropy)
    
if __name__ == "__main__":
    main()


