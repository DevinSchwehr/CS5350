import torch
import numpy as np
import os
from torch import nn
import torch.nn.functional as functional

class Neural_Relu_Network(nn.Module):
    def __init__(self,input_len,width,weights,bias):
        self.weights = weights
        self.bias = bias
        super(Neural_Relu_Network,self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_len,width),
            nn.ReLU(),
            nn.Linear(width,width),
            nn.ReLU(),
            nn.Linear(width,width),
            nn.ReLU(),
            nn.Linear(width,2)
        )

    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def model(self,x):
        return x @ self.weights.t() + self.bias

class Neural_Tanh_Network(nn.Module):
    def __init__(self,input_len,width,weights,bias):
        self.weights = weights
        self.bias = bias
        super(Neural_Tanh_Network,self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_len,width),
            nn.Tanh(),
            nn.Linear(width,width),
            nn.Tanh(),
            nn.Linear(width,width),
            nn.Tanh(),
            nn.Linear(width,2)
        )

    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def model(self,x):
        return x @ self.weights.t() + self.bias

def Adjust_Labels(data):
    for example in data:
        if example[len(example)-1] == 0:
            example[len(example)-1] = -1
    return data

def Get_Error(logits,outputs):
    error = 0
    for i in range(len(outputs)):
        if torch.sign(logits[i]) != outputs[i]:
            error +=1
    return (error/len(outputs))*100

def Get_Accuracy(data,model,loss_fn):
        inputs = []
        outputs = []
        for example in data:
            inputs.append(example[slice(0,len(example)-1)])
            outputs.append(example[len(example)-1])
        # print(model)
        input_tensor = torch.from_numpy(np.array(inputs))
        output_tensor = torch.from_numpy(np.array(outputs))
        # test_output_tensor = torch.from_numpy(np.array(test_outputs))
        pred = model(input_tensor)
        test_loss = loss_fn(pred,output_tensor.long()).item()
        correct = (pred.argmax(1) == output_tensor).type(torch.float).sum().item()
        return (correct/len(data))*100


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(here,'train.csv')
    test_file = os.path.join(here,'test.csv')
    data = np.genfromtxt(train_file,delimiter=',',dtype=np.float32)
    test_data = np.genfromtxt(test_file,delimiter=',',dtype=np.float32)

    # data = Adjust_Labels(data)
    # test_data = Adjust_Labels(test_data)
    test_outputs = []
    for test_example in test_data:
        test_outputs.append(test_example[len(test_example)-1])
    widths = [5,10,15,25,50,100]
    gamma = 0.01
    d = 0.01
    for width in widths:
        print("For width: "+ str(width))
        # width = int(input("Please enter the width\n"))
        weights = torch.randn(1,width,requires_grad=True)
        bias = torch.randn(1,requires_grad=True)
        # model = Neural_Relu_Network(len(data[0])-1,width,weights,bias)
        model = Neural_Tanh_Network(len(data[0])-1,width,weights,bias)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(1,100):
            np.random.shuffle(data)
            inputs = []
            outputs = []
            for example in data:
                inputs.append(example[slice(0,len(example)-1)])
                outputs.append(example[len(example)-1])
            # print(model)
            input_tensor = torch.from_numpy(np.array(inputs))
            output_tensor = torch.from_numpy(np.array(outputs))
            learning_rate = gamma/(1+(gamma/d)*epoch)
            optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

            pred = model(input_tensor)
            loss = loss_fn(pred,output_tensor.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print("Training Accuracy " + str(Get_Accuracy(data,model,loss_fn)))
        print("Test Accuracy " + str(Get_Accuracy(test_data,model,loss_fn)))
    

if __name__ == "__main__":
    main()


