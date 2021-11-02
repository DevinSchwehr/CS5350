import numpy as np
import csv
import os

from numpy.lib.function_base import percentile

def Perceptron(data,weight_vector):
    weight_vector = np.zeros(4)
    learning_rate = 0.2
    for example in data:
        inputs = example[slice(0,len(example)-1)]
        prediction = np.sign(np.dot(weight_vector,inputs))
        label = example[len(example)-1]
        if (prediction <= 0 and label > 0) or (prediction > 0 and label < 0):
            weight_vector = np.add(weight_vector,learning_rate*(label*inputs))
    return weight_vector

def Voted_Perceptron(data,weight_vector):
    weight_dict = {}
    vote_dict = {}
    m = 0
    learning_rate = 0.2
    weight_dict[m] = weight_vector
    vote_dict[m] = 0
    for example in data:
        inputs = example[slice(0,len(example)-1)]
        label = example[len(example)-1]
        output = label*np.dot(weight_vector,inputs)
        if output <= 0:
            new_weight = weight_vector + learning_rate*(label*inputs)
            m +=1
            weight_dict[m] = new_weight
            vote_dict[m] = 1
            weight_vector = np.copy(new_weight)
        else:
            vote_dict[m] = vote_dict[m]+1
    return weight_dict,vote_dict

def Average_Perceptron(data,weight_vector,average_weight):
    learning_rate = 0.2
    for example in data:
        inputs = example[slice(0,len(example)-1)]
        label = example[len(example)-1]
        output = label*np.dot(weight_vector,inputs)
        if output <= 0:
            weight_vector = weight_vector + learning_rate*(label*inputs)
        average_weight = average_weight + weight_vector
    return weight_vector,average_weight

def Prediction_Error(weight_vector,test_data):
    error = 0
    for example in test_data:
        inputs = example[slice(0,len(example)-1)]
        prediction = np.sign(np.dot(weight_vector,inputs))
        if (prediction <= 0 and example[len(example)-1] > 0) or (prediction > 0 and example[len(example)-1] == 0):
            error += 1
    return error/len(test_data)

def Voted_Prediction_Error(data,weight_dict,vote_dict):
    error = 0
    for example in data:
        inputs = example[slice(0,len(example)-1)]
        output = example[len(example)-1]
        summation = 0
        for key in weight_dict:
            summation += vote_dict[key]*np.sign(np.dot(weight_dict[key],inputs))
        prediction = np.sign(summation)
        if (prediction <= 0 and output == 1) or (prediction > 0 and output == 0):
            error +=1
    return error/len(data)

def Adjust_Labels(data):
    for example in data:
        if example[len(example)-1] == 0:
            example[len(example)-1] = -1
    return data

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(here,'train.csv')
    test_file = os.path.join(here,'test.csv')
    data = np.genfromtxt(train_file,delimiter=',',dtype=np.float32)
    test_data = np.genfromtxt(test_file,delimiter=',',dtype=np.float32)
    data = Adjust_Labels(data)
    test_data = Adjust_Labels(test_data)
    weight_vector = np.zeros(4)
    prediction_error = 0
    algorithm = int(input('Type 1 for Perceptron, 2 for Voted Perceptron, and 3 for Average Perceptron \n'))
    if algorithm == 1:
        for index in range (0,10):
            weight_vector = Perceptron(data,weight_vector)
            prediction_error += Prediction_Error(weight_vector,test_data)
        prediction_error /= 10
        print('Final Weight Vector ' + str(weight_vector))
        print ('Average Prediction Error ' + str(prediction_error))
    if algorithm == 2:
        weight_dict = {}
        vote_dict = {}
        for index in range(0,10):
            weight_dict,vote_dict = Voted_Perceptron(data,weight_vector)
            prediction_error += Voted_Prediction_Error(test_data,weight_dict,vote_dict)
        prediction_error /= 10
        for key in weight_dict:
            print('Weight Vector '+ str(weight_dict[key]) + ' Votes ' + str(vote_dict[key]))
        print('Average Test Error ' + str(prediction_error))
    if algorithm == 3:
        average_weight = np.zeros(4)
        for index in range(0,10):
            weight_vector,average_weight = Average_Perceptron(data,weight_vector,average_weight)
            prediction_error += Prediction_Error(average_weight,test_data)
        prediction_error /= 10
        print('Final Average Weight Vector ' + str(weight_vector))
        print('Average Prediction Error ' + str(prediction_error))


if __name__ == "__main__":
    main()