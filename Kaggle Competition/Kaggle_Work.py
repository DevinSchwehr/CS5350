from os import write
import numpy as np
import pandas as pd
from sklearn import base, tree
import sklearn
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import sklearn.preprocessing as prep
import csv
import torch
from torch import nn
import torch.nn.functional as functional

def Remove_Unknown_Values(data, column_names):
    for column in column_names:
        if '?' in pd.unique(data[column]):
            value_counts = data[column].value_counts()
            value_counts = value_counts.drop(labels=['?'])
            most_common_value = value_counts.idxmax()
            data[column].replace({'?': most_common_value}, inplace=True)

def Improved_Remove_Unknown(data,column_names):
    for column in column_names:
        if '?' in pd.unique(data[column]):
            for i in range(0,data.shape[0]):
                if data.at[i,column] == '?':
                    filter = data["income>50K"] == data.iloc[i]["income>50K"]
                    filtered_data = data[filter]
                    value_counts = filtered_data[column].value_counts()
                    value_counts = value_counts.drop(labels=['?'])
                    most_common_value = value_counts.idxmax()
                    data.at[i,column] = most_common_value


def Perform_Decision_Tree(dec_tree,encoded_train_data,encoded_test_data,train_labels):
    dec_tree.fit(encoded_train_data,train_labels)

    test_result = dec_tree.predict(encoded_test_data)

    Write_To_CSV('decision_tree_results.csv',test_result)

def Perform_AdaBoost(encoded_train_data,encoded_test_data,train_labels):
    ada = AdaBoostClassifier(n_estimators=500,random_state=1, base_estimator= tree.DecisionTreeClassifier(max_depth=1))
    ada.n_features_in_ = 900
    ada.fit(encoded_train_data,train_labels)
    results = ada.predict(encoded_test_data)
    score = ada.score(encoded_train_data,train_labels)
    Write_To_CSV('AdaBoost_Results.csv',results)

def Perform_Linear_Regression(encoded_train_data,encoded_test_data,train_labels):
    regressor = LinearRegression()
    regressor.n_features_in_ = 900
    regressor.fit(encoded_train_data,train_labels)
    print('Training Score: ' + str(regressor.score(encoded_train_data,train_labels)))
    results = regressor.predict(encoded_test_data)
    Write_To_CSV('Linear-Regression-Results.csv',results)

def Perform_Perceptron(encoded_train_data,encoded_test_data,train_labels):
    perceptron = Perceptron(max_iter = 10000, eta0=0.1, random_state=0)
    perceptron.fit(encoded_train_data,train_labels)
    results = perceptron.predict(encoded_test_data)
    Write_To_CSV('Perceptron-Results.csv',results)

def Perform_Bagging(encoded_train_data,encoded_test_data,train_labels):
    #Default to Decision Tree Classifier
    bagging = BaggingClassifier(n_estimators=500,random_state=1).fit(encoded_train_data,train_labels)
    results = bagging.predict(encoded_test_data)
    Write_To_CSV('Bagged_Trees.csv', results)

def Perform_Scikit_Neural(encoded_train_data,encoded_test_data,train_labels):
    nn = MLPClassifier(hidden_layer_sizes=50,max_iter=100,random_state=1)
    nn.fit(encoded_train_data,train_labels)
    print('Training Score: ' + str(nn.score(encoded_train_data,train_labels)))
    results = nn.predict(encoded_test_data)
    Write_To_CSV("Neural_Network_Results",results)

def Perform_Naive_Bayes(encoded_train_data,encoded_test_data,train_labels):
    bayes = GaussianNB()
    bayes.fit(encoded_train_data,train_labels)
    print('Training Score: ' + str(bayes.score(encoded_train_data,train_labels)))
    results = bayes.predict(encoded_test_data)
    Write_To_CSV("Naive_Bayes",results)

def Write_To_CSV(filename, results):
    with open(filename,'w',newline='') as fh:
        writer = csv.writer(fh,delimiter = ',')
        writer.writerow(['ID','Prediction'])
        writer.writerows(enumerate(results,1))
    print("got output for" + filename)


def main():
    train_data = pd.read_csv(r"C:\Users\devin\OneDrive\Documents\CS5350\CS5350\Kaggle Competition\train_final.csv")
    test_data = pd.read_csv(r"C:\Users\devin\OneDrive\Documents\CS5350\CS5350\Kaggle Competition\test_final.csv")
    cols = ['age','workclass','fnlwgt','education','education.num','marital.status','occupation','relationship','race','sex','capital.gain','capital.loss','hours.per.week','native.country']
    test_cols = cols.copy()
    Improved_Remove_Unknown(train_data,cols)
    Remove_Unknown_Values(test_data,test_cols)
    test_data = test_data.drop(columns='ID')
    train_labels = train_data['income>50K']
    # train_labels = train_labels.replace(0,-1)
    train_data = train_data.drop(columns='income>50K')

    std = prep.StandardScaler(with_mean=False)
    encoder = prep.OneHotEncoder(handle_unknown='ignore')
    encoder.fit(train_data)
    encoded_train_data = encoder.transform(train_data)
    encoded_test_data = encoder.transform(test_data)
    std.fit(encoded_train_data)
    encoded_train_data = std.transform(encoded_train_data)
    encoded_test_data = std.transform(encoded_test_data)

    # dec_tree = tree.DecisionTreeClassifier(criterion='entropy')
    # Perform_Decision_Tree(dec_tree,encoded_train_data,encoded_test_data,train_labels)

    # Perform_AdaBoost(encoded_train_data,encoded_test_data,train_labels)

    Perform_Linear_Regression(encoded_train_data,encoded_test_data,train_labels)

    # Perform_Perceptron(encoded_train_data,encoded_test_data,train_labels)

    # Perform_Scikit_Neural(encoded_train_data,encoded_test_data,train_labels)

    # Perform_Bagging(encoded_train_data,encoded_test_data,train_labels)

    # Perform_Naive_Bayes(encoded_train_data.toarray(),encoded_test_data.toarray(),train_labels)

if __name__ == "__main__":
    main()
