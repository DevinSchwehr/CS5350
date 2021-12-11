from typing import NewType
import numpy as np;
import os;
import random

from numpy.core.numeric import zeros_like;


def NN_Initialization(input_length, width):
    # Get the length of one input example
    # input_length = len(inputs)
    neural_net = list()
    #Create the first Hidden layer, with (width) neurons each with a number of connections
    #for each input plus the bias
    first_hidden = [[np.random.normal() for i in range(input_length+1)]for i in range(width)]
    #Repeat process for second hidden layer, but with number of connections for previous
    #hidden layer plus bias
    second_hidden = [[np.random.normal() for i in range(width+1)]for i in range(width)]
    #Now we create the output layer, though there is only one neuron
    # output_layer = [[random() for i in range(width+1)]for i in range(len(outputs))]
    #Need two y values
    output_layer = [[np.random.normal() for i in range(width+1)]]

    # # input_length = len(inputs)
    # neural_net = list()
    # #Create the first Hidden layer, with (width) neurons each with a number of connections
    # #for each input plus the bias
    # first_hidden = [[0 for i in range(input_length+1)]for i in range(width)]
    # #Repeat process for second hidden layer, but with number of connections for previous
    # #hidden layer plus bias
    # second_hidden = [[0 for i in range(width+1)]for i in range(width)]
    # #Now we create the output layer, though there is only one neuron
    # # output_layer = [[random() for i in range(width+1)]for i in range(len(outputs))]
    # #Need two y values
    # output_layer = [[0 for i in range(width+1)]]


    neural_net.append(first_hidden)
    neural_net.append(second_hidden)
    neural_net.append(output_layer)
    return neural_net

def Sigmoid_Activation(input_vec,weight_vec):
    #Sum up the weights*inputs
    sum = 0
    for i in range(len(weight_vec)-1):
        sum += weight_vec[i]*input_vec[i]
    #Now add in the bias at the end
    sum += weight_vec[len(weight_vec)-1]*1
    sigmoid = 1.0 / (1.0 + np.exp(-sum))
    return sigmoid

def Get_Y(input_vec,weight_vec):
    sum = 0
    for i in range(len(weight_vec)-1):
        sum += weight_vec[i]*input_vec[i]
    #Now add in the bias at the end
    sum += weight_vec[len(weight_vec)-1]*1
    return sum

def Forward_Pass(neural_net,inputs):
    #We will store the z values for each layer as an array and put it in z_layers
    z_layers = []
    current_inputs = inputs
    for i in range(len(neural_net)-1):
        z_results = []
        for neuron_weights in neural_net[i]:
            z_val = Sigmoid_Activation(current_inputs,neuron_weights)
            z_results.append(z_val)
        z_layers.append(z_results)
        current_inputs = z_results
    y_val = Get_Y(current_inputs,neural_net[len(neural_net)-1][0])
    z_layers.append([y_val])
    return z_layers

def Back_Propagation(neural_net,z_layers,expected,inputs):
    #Prepare our partial derivatives
    # partials = [[]] * len(neural_net)
    weight_partials = [None]*3
    z_partials = [None]*3
    # Get dy/dl
    dy = z_layers[len(z_layers)-1][0] - expected
    #Calculate the top partials, those that are dy/dl * wi
    for i in reversed(range(len(neural_net))):
        #Check if we are at the top layer
        if i == len(neural_net)-1:
            top_partials = []
            for z in z_layers[i-1]:
                top_partials.append(z*dy)
            top_partials.append(dy)
            # weight_partials.append(top_partials)
            weight_partials[i] = [top_partials]
        #Otherwise check if we are in the middle layer
        if i == len(neural_net)-2:
            mid_partials = []
            mid_z_partials = []
            #iterate over each set of weights
            for j in range(len(neural_net[i])):
                curr_z_partial = dy*neural_net[i+1][0][j]*Derivative_Function(z_layers[i][j])
                weight_set_partials = []
                for k in range(len(neural_net[i][j])-1):
                    #If it is a z node
                    weight_set_partials.append(curr_z_partial*z_layers[i-1][k])
                #Since the bias is always 1, just append the z partial to the end
                weight_set_partials.append(curr_z_partial)
                mid_partials.append(weight_set_partials)
                mid_z_partials.append(curr_z_partial)
            weight_partials[i] = mid_partials
            z_partials[i] = mid_z_partials
        #If we aren't at the top or middle, we're at the bottom
        if i == 0:
            bottom_partials = []
            bottom_z_partials = []
            for j in range(len(neural_net[i])):
                current_z_partials = []
                # curr_partial = 0
                #get the weight partials
                for k in range(len(z_partials[i+1])):
                    zp = z_partials[i+1][k] * neural_net[i+1][k][j] * Derivative_Function(z_layers[i][j])
                    # curr_partial += zp*inputs[j]
                    current_z_partials.append(zp)
                bottom_z_partials.append(current_z_partials)
                #Now take these z partials and apply them to the weight set
                curr_bottom_partials = []
                for k in range(len(neural_net[i][j])-1):
                    curr_bottom_partials.append(np.sum(np.multiply(current_z_partials,inputs[k])))
                curr_bottom_partials.append(np.sum(current_z_partials))
                bottom_partials.append(curr_bottom_partials)
            weight_partials[i] = bottom_partials
    return weight_partials

def Update_Weights(neural_net,gradients,learning_rate):
    for i in range(len(neural_net)):
        for j in range(len(neural_net[i])):
            for k in range(len(neural_net[i][j])):
                neural_net[i][j][k] -= gradients[i][j][k]*learning_rate
    return neural_net            

def Derivative_Function(z_val):
    return z_val * (1-z_val)

def Adjust_Labels(data):
    for example in data:
        if example[len(example)-1] == 0:
            example[len(example)-1] = -1
    return data

def Calculate_Error_Percent(nn,data):
    error = 0
    for example in data:
        x = example[slice(0,len(example)-1)]
        y = example[len(example)-1]
        z_vals = Forward_Pass(nn,x)
        if np.sign(z_vals[len(z_vals)-1][len(z_vals[len(z_vals)-1])-1]) != y:
            error += 1
    return (error/len(data))*100

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(here,'train.csv')
    test_file = os.path.join(here,'test.csv')
    data = np.genfromtxt(train_file,delimiter=',',dtype=np.float32)
    test_data = np.genfromtxt(test_file,delimiter=',',dtype=np.float32)
    data = Adjust_Labels(data)
    test_data = Adjust_Labels(test_data)
    test_nn = [[[-2,-3,-1],[2,3,1]],[[-2,-3,-1],[2,3,1]],[[2,-1.5,-1]]]
    width = int(input("Please input the desired width of each layer \n"))
    nn = None
    nn = NN_Initialization(len(data[0])-1,width)
    for epoch in range(1,25):
        gamma = 0.01
        d = 0.01
        np.random.shuffle(data)
        for example in data:
            x = example[slice(0,len(example)-1)]
            y = example[len(example)-1]
            z_vals = Forward_Pass(nn,x)
            gradient_vals = Back_Propagation(nn,z_vals,y,x)
            learning_rate = gamma/(1+(gamma/d)*epoch)
            nn = Update_Weights(nn,gradient_vals,learning_rate)
    print("Training Error: " + str(Calculate_Error_Percent(nn,data)))
    print("Test Error: " + str(Calculate_Error_Percent(nn,test_data)))

if __name__ == "__main__":
    main()