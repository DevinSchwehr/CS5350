import numpy as np

def Calculate_Cost(data,weights,y):
    i = 0
    sum = 0
    while i < len(data):
        # print(str(i))
        curr_y = y[i]
        vector_mult = Calculate_Vector_Mult(data[i],weights)
        sum += np.square(curr_y-vector_mult)
        i += 1
    return sum * 1/2

def Calculate_Vector_Mult(row,weights):
    sum = 0
    i = 0
    while i < len(weights):
        sum += row[i] * weights[i]
        i+=1
    return sum

def main():
 x_matrix = np.matrix([[1,-1,2,1],[1,1,3,1],[-1,1,0,1],[1,2,-4,1],[3,-1,-1,1]])
 w1 = np.matrix([0.1,-0.1,0.2,0.1])
 w2 = np.matrix([0.43,0.23,1.19,0.43])
 w3 = np.matrix([0.553,0.107,1.19,0.307])
 w4 = np.matrix([0.7216,0.4392,0.5156,0.4756])
 w5 = np.matrix([0.216,-0.61,0.68,0.307])
 y_matrix = np.matrix([1,4,-1,-2,0])

 x_arr = [[1,-1,2,1],[1,1,3,1],[-1,1,0,1],[1,2,-4,1],[3,-1,-1,1]]
 w1 = [0.1,-0.1,0.2,0.1]
 w2 = [0.43,0.23,1.19,0.43]
 w3 = [0.553,0.107,1.19,0.307]
 w4 = [0.7216,0.4392,0.5156,0.4756]
 w5 = [0.216,-0.61,0.68,0.307]
 y_arr = [1,4,-1,-2,0]
 print(Calculate_Cost(x_arr,w1,y_arr))
 print(Calculate_Cost(x_arr,w2,y_arr))
 print(Calculate_Cost(x_arr,w3,y_arr))
 print(Calculate_Cost(x_arr,w4,y_arr))
 print(Calculate_Cost(x_arr,w5,y_arr))


if __name__ == "__main__":
    main()