import numpy as np
import random

def Sigmoid_Activation(val):
    sigmoid = 1.0 / (1.0 + np.exp(-val))
    return sigmoid

def main():
    x = np.array([[0.5,-1,0.3],[-1,-2,-2],[1.5,0.2,-2.5]])
    y = np.array([1,-1,1])
    learning_rates = [0.001,0.005,0.0025]
    weights = np.array([np.random.normal() for i in range(3)])
    for learning_rate in learning_rates:
        i = random.randint(0,2)
        gradient = -(1-Sigmoid_Activation(y[i]*np.dot(weights,x[i])))*y[i]*x[i] - weights[i]
        print(str(gradient))
        for j in range(len(weights)):
            weights -= gradient * learning_rate

if __name__ == "__main__":
    main()