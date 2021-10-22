import numpy as np
import pandas as pd
import operator
import csv 
import random
import matplotlib.pyplot as plt

# data = pd.DataFrame
# test_data = pd.DataFrame
train_data = []
test_data = []


def Calculate_Gradient(data,weights):
    i = 0
    j = 0
    sum = 0
    gradient = []
    while j < len(weights):
        sum = 0
        i = 0
        while i < len(data):
            y = data[i][len(data[i])-1]
            vector_mult = Calculate_Vector_Mult(data[i],weights)
            sum += (y-vector_mult)*data[i][j]
            i += 1
        gradient.append(sum * -1)
        j +=1
    return gradient

def Calculate_Cost(data,weights):
    i = 0
    sum = 0
    while i < len(data):
        # print(str(i))
        y = data[i][len(data[i])-1]
        vector_mult = Calculate_Vector_Mult(data[i],weights)
        sum += np.square(y-vector_mult)
        i += 1
    return sum * 1/2

def Calculate_Vector_Mult(row,weights):
    sum = 0
    i = 0
    while i < len(weights):
        sum += row[i] * weights[i]
        i+=1
    return sum

def Calc_Norm(weights,new_weights):
    diff = list(map(operator.sub, new_weights,weights))
    return np.linalg.norm(diff)

def Batch_Descent(train_data,test_data):
    weights=  [0,0,0,0,0,0,0]
    costs = []
    steps = []
    r = 0.01
    tolerance = 1
    j = 0
    while tolerance > 0.000001:
        # print('at iteration ' + str(j) + '\n')
        cost = Calculate_Cost(train_data,weights)
        costs.append(cost)
        # print('with weights ' + str(weights) + '  cost = ' + str(cost))
        # print('cost: ' + str(cost) + '\n')
        gradient = Calculate_Gradient(train_data,weights)
        new_weights = []
        i = 0
        while i < len(weights):
            new_weights.append(weights[i] - (r*gradient[i]))
            i +=1
        tolerance = Calc_Norm(weights,new_weights)
        # print('new tolerance is ' + str(tolerance) + '\n')
        weights = new_weights
        steps.append(j)
        j+=1
    print('final cost is ' + str(Calculate_Cost(train_data,weights)) + '\n')
    print('test cost is ' + str(Calculate_Cost(test_data,weights)) + '\n')
    print('Final Weight Vector: ' + str(weights))
    steps.append(j)
    costs.append(Calculate_Cost(train_data,weights))
    Create_Graph(costs,steps,'Batch Costs')

def Stochastic_Descent(train_data,test_data):
    weights = [0,0,0,0,0,0,0]
    new_weights = [0,0,0,0,0,0,0]
    r = 0.01
    tolerance = 1
    j = 0
    step = 0
    steps = []
    costs = []
    while tolerance > 0.000001:
        steps.append(step)
        cost = Calculate_Cost(train_data,weights)
        costs.append(cost)
        new_weights = []
        training_sample = train_data[random.randint(0,len(train_data)-1)]
        j = 0
        mult_vector = Calculate_Vector_Mult(training_sample,weights)
        y = training_sample[len(training_sample)-1]
        while j < len(weights):
            # weights[j] = weights[j] + r*(y-mult_vector)*train_data[i][j]
            new_weights.append(weights[j] + r*(y-mult_vector)*training_sample[j])
            j+=1
        tolerance = Calc_Norm(weights,new_weights)
        weights = new_weights
        step +=1
    print('final cost is ' + str(Calculate_Cost(train_data,weights)) + '\n')
    print('test cost is ' + str(Calculate_Cost(test_data,weights)) + '\n')
    print('Final Weight Vector: ' + str(weights))
    steps.append(step)
    costs.append(Calculate_Cost(train_data,weights))
    Create_Graph(costs,steps,"Stochastic Costs")

def Analytical_Best(train_data):
    x_rows = []
    y_rows = []
    for row in train_data:
        i = 0
        current_row = []
        while i < len(row)-1:
            current_row.append(row[i])
            i+=1
        y_rows.append(row[len(row)-1])
        x_rows.append(current_row)
    x_matrix = np.matrix(x_rows)
    y_matrix = np.matrix(y_rows)
    inverted_x = np.linalg.inv(x_matrix.transpose()*x_matrix)
    xy_matrix = y_matrix*x_matrix
    weight_vector = xy_matrix*inverted_x
    print(str(weight_vector) + '\n')

def Create_Graph(y,x,title):
    plt.plot(x,y, label="Cost")
    plt.xlabel('Steps')
    plt.ylabel('Cost')
    plt.title(title)
    plt.legend()
    plt.show()

def main():
    training_file = r"./train.csv"
    test_file = r"./test.csv"

    with open(training_file, newline='')as f:
        reader = csv.reader(f)
        train_data = list(reader)
    with open(test_file, newline='')as f:
        reader = csv.reader(f)
        test_data = list(reader)

    i = 0
    j = 0
    while i < len(train_data):
        j = 0
        while j < len(train_data[i]):
            train_data[i][j] = float(train_data[i][j])
            j+=1
        i+=1

    i = 0
    j = 0
    while i < len(test_data):
        j = 0
        while j < len(test_data[i]):
            test_data[i][j] = float(test_data[i][j])
            j+=1
        i+=1
    
    print('Analytical Weight Vector: \n')
    Analytical_Best(train_data)
    decider = input("Please input what type of Regression you would like to perform\n")
    if(decider == "Batch"):
        Batch_Descent(train_data,test_data)
    if(decider == "Stochastic"):
        Stochastic_Descent(train_data,test_data)


if __name__ == "__main__":
    main()
