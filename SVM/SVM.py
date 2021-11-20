import numpy as np
import os
from scipy.optimize._minimize import minimize

gaussian_value = 0

def Adjust_Labels(data):
    for example in data:
        if example[len(example)-1] == 0:
            example[len(example)-1] = -1
    return data

def Primal_Sub_Gradient(data,weights,yt,c):
    N = len(data)
    for example in data:
        inputs = example[0:4]
        inputs = np.append(inputs,1)
        output = example[len(example)-1]
        if output*np.dot(weights,inputs) <= 1:
            zero_bias_weights = weights[0:4]
            zero_bias_weights = np.append(zero_bias_weights,0)
            left = weights - yt*zero_bias_weights
            right = yt*c*N*output*inputs
            weights = left+right
        else:
            weights[0:4] = (1-yt)*weights[0:4] 
    return weights

#This method is used by the dual function that does not use the Kernel. 
def objective(a, inputs,outputs):
    alpha_sum = np.sum(a)
    x_mat = np.asmatrix(inputs)
    x_matrix = x_mat * np.transpose(x_mat)
    ay_mat = np.outer(a,outputs)
    ay_matrix = ay_mat*np.transpose(ay_mat)
    sum = np.sum(ay_matrix*x_matrix)
    return (1/2)*(sum - alpha_sum)

#This objective follows the same format as the objective above, but instead of using X^T*X, it uses the Kernel method.
def Gaussian_Objective(a,inputs,outputs,rate):
    alpha_sum = np.sum(a)
    alpha_matrix = np.outer(a,np.transpose(a))
    y_matrix = np.outer(outputs,np.transpose(outputs))
    sum = np.sum(alpha_matrix*y_matrix*gaussian_value)
    return (1/2)*sum-alpha_sum

#The primary Dual method, this one calls the minimization function.
def Dual_SVM(inputs,outputs,c,a):
    bound = [(0,c)]*len(a)
    cons = [{'type':'eq','fun': lambda a: np.dot(a,outputs),'jac': lambda a: outputs}]
    sol = minimize(fun=objective, x0=a, args=(inputs,outputs), method='SLSQP', constraints=cons, bounds=bound)
    return sol

#Same as above, but calls the Gaussian Objective
def Gaussian_Dual_SVM(inputs,outputs,c,a,rate):
    bound = [[0,c]]*len(a)
    cons = [{'type':'eq','fun': lambda a: np.dot(a,outputs),'jac': lambda a: outputs}]
    sol = minimize(fun=Gaussian_Objective, x0=a, args=(inputs,outputs,rate), method='SLSQP', constraints=cons, bounds=bound)
    return sol

#A kernel function that vectorizes the matrices in order to avoid iterating over them to get the matrix.
def Gaussian_Kernel(xi,xj,rate):
    xi = np.asmatrix(xi)
    xj = np.asmatrix(xj)
    first_mat = np.sum(np.multiply(xi,xi),axis=1)
    xt = np.transpose(xj)
    second_mat = np.sum(np.multiply(xt,xt),axis=0)
    x_mat = xi*np.transpose(xj)
    right = (2*x_mat)
    return np.exp(-(first_mat+second_mat-right)/rate)

#This is the prediction error method utilized by both the primal and first dual SVM implementations.
def Prediction_Error(weight_vector,test_data):
    error = 0
    for example in test_data:
        inputs = example[slice(0,len(example)-1)]
        prediction = np.sign(np.dot(weight_vector[0:4],inputs)+weight_vector[4])
        if prediction != example[len(example)-1]:
            error += 1
    return error/len(test_data)

#This prediction error method supports the Gaussian dual implementation.
def Gaussian_Prediction_Error(alphas,outputs,inputs,supports,rate):
    error = 0
    for index in range(len(outputs)):
        example = inputs[index]
        #Avoid out of bounds errors.
        if(len(alphas) < len(outputs)):
            ay = np.sum(alphas[i]*outputs[i] for i in range(len(alphas)))
        else:
            ay = np.sum(alphas[i]*outputs[i] for i in range(len(outputs)))
            
        prediction = np.sign(np.sum(ay*Gaussian_Kernel(supports[i],example,rate) for i in range(len(alphas))))
        if prediction != outputs[index]:
            error += 1
    return error/len(outputs)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(here,'train.csv')
    test_file = os.path.join(here,'test.csv')
    data = np.genfromtxt(train_file,delimiter=',',dtype=np.float32)
    test_data = np.genfromtxt(test_file,delimiter=',',dtype=np.float32)
    data = Adjust_Labels(data)
    test_data = Adjust_Labels(test_data)
    option = int(input("Enter 1 for Primal SVM, 2 for Dual SVM, 3 for Gaussian SVM \n"))

    #If performing primal implementation
    if(option == 1):
        c_arr = [100/873,500/873,700/873]
        for iteration in range(0,3):
            c = c_arr[iteration]
            weight_vector = np.zeros(5)
            test_prediction_error = 0
            train_prediction_error = 0
            for index in range(1,101):
                np.random.shuffle(data)
                learning_rate = 0.1
                a = 1
                #These are the two yt functions. Pick which one you would like to utilize by un-commenting one and commenting out the other.
                # yt = learning_rate/((1+(learning_rate/a)*index))
                yt = learning_rate/((1+index))
                weight_vector = Primal_Sub_Gradient(data,weight_vector,yt,c)
                test_prediction_error += Prediction_Error(weight_vector,test_data)
                train_prediction_error += Prediction_Error(weight_vector, data)
            test_prediction_error /= 100
            train_prediction_error /= 100
            print("For C value: " + str(c))
            print(str(weight_vector))
            #Multiply by 100 in order to get the percentages
            print("Train Error: " + str(train_prediction_error*100))
            print("test error: " + str(test_prediction_error*100))

    #Dual SVM implementation.
    if(option == 2):
        c_arr = [100/873,500/873,700/873]
        alphas = np.random.rand(len(data))
        bias_data = np.insert(data,4,1,axis=1)
        inputs = []
        outputs = []
        #Set up the data, insert the bias into the weight vector.
        for example in bias_data:
            inputs.append(example[0:5])
            outputs.append(example[len(example)-1])
        for iteration in range(0,3):
            c = c_arr[iteration]
            mini = Dual_SVM(inputs,outputs,c,alphas)
            filter_arr = []
            #Filter the alpha values.
            for alpha in mini.x:
                if alpha > 0.001:
                    filter_arr.append(True)
                else:
                    filter_arr.append(False)
            result = mini.x[filter_arr]
            #Create the weight vector
            weight_vector = np.sum(result[i]*inputs[i]*outputs[i] for i in range(len(result)))
            print(str(weight_vector))
            train_prediction_error = Prediction_Error(weight_vector,data)
            print(train_prediction_error)

    #Gaussian Dual Implementation
    if(option ==3):
        data = data[:int(len(data)/2)]
        test_data = test_data[:int(len(test_data)/2)]
        rate_arr = [0.1,0.5,1,5,100]
        c_arr = [100/873,500/873,700/873]
        alphas = np.random.rand(len(data))
        inputs = []
        outputs = []
        #Gather the X vectors and y values
        for example in data:
            inputs.append(example[0:4])
            outputs.append(example[len(example)-1])
        prev = []
        for rate in rate_arr:
            for iteration in range(0,3):
                c = c_arr[iteration]
                #Get the kernel value now, saves us runtime calculations.
                global gaussian_value 
                gaussian_value = Gaussian_Kernel(inputs,inputs,rate)
                mini = Gaussian_Dual_SVM(inputs,outputs,c,alphas,rate)
                filter_arr = []
                #Filter the resulting alphas
                for alpha in mini.x:
                    if  alpha > 0.1:
                        filter_arr.append(True)
                    else:
                        filter_arr.append(False)
                result = mini.x[filter_arr]

                train_prediction_error = Gaussian_Prediction_Error(result,outputs,inputs,inputs,rate)

                #Set up the data for the test data
                test_inputs = []
                test_outputs = []
                for example in test_data:
                    test_inputs.append(example[0:4])
                    test_outputs.append(example[len(example)-1])
                test_prediction_error = Gaussian_Prediction_Error(result,test_outputs,test_inputs,inputs,rate)
                print("for rate of: " + str(rate) + " and C value of: " + str(c))
                print("Training Error: " + str(train_prediction_error))
                print("Test Error: " + str(test_prediction_error))
                print("Number of Support vectors: " + str(len(result)))
                print("# of Overlapping Supports: " + str(len(np.intersect1d(result,prev))))
                prev = result


if __name__ == "__main__":
    main()