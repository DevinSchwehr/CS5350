from os import write
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
import sklearn
import sklearn.preprocessing as prep
import csv

def Remove_Numeric_Values(data, column_names):
    for column in column_names:
        if data.dtypes[column] == np.int64:
            median_value = np.median(data[column].values)
            # i = 0
            # while i < len(data[column].values):
            #     if int(data[column].iloc[i]) < median_value:
            #         data[column].iloc[i] = '-'
            #     else:
            #         data[column].iloc[i] = '+'
            #     i += 1
            data.loc[data[column] >= median_value, column] = '+'
            data.loc[data[column] != '+', column] = '-'

def Remove_Unknown_Values(data, column_names):
    for column in column_names:
        if '?' in pd.unique(data[column]):
            value_counts = data[column].value_counts()
            value_counts = value_counts.drop(labels=['?'])
            most_common_value = value_counts.idxmax()
            data[column].replace({'?': most_common_value}, inplace=True)

def Perform_Decision_Tree(dec_tree,encoded_train_data,encoded_test_data,train_labels):
    dec_tree.fit(encoded_train_data,train_labels)

    test_result = dec_tree.predict(encoded_test_data)

    Write_To_CSV('decision_tree_results.csv',test_result)

def Perform_AdaBoost(encoded_train_data,encoded_test_data,train_labels):
    ada = AdaBoostClassifier(n_estimators=100,random_state=0)
    ada.fit(encoded_train_data,train_labels)
    results = ada.predict(encoded_test_data)
    Write_To_CSV('AdaBoost_Results.csv',results)

def Write_To_CSV(filename, results):
    with open(filename,'w',newline='') as fh:
        writer = csv.writer(fh,delimiter = ',')
        writer.writerow(['ID','Predictions'])
        writer.writerows(enumerate(results,1))
    print("got output for" + filename)


def main():
    train_data = pd.read_csv(r"C:\Users\devin\OneDrive\Documents\CS5350\CS5350\Kaggle Competition\train_final.csv")
    test_data = pd.read_csv(r"C:\Users\devin\OneDrive\Documents\CS5350\CS5350\Kaggle Competition\test_final.csv")
    cols = ['age','workclass','fnlwgt','education','education.num','marital.status','occupation','relationship','race','sex','capital.gain','capital.loss','hours.per.week','native.country']
    test_cols = cols.copy()
    # test_cols.remove('income>50K')
    # Remove_Numeric_Values(train_data,cols)
    # Remove_Numeric_Values(test_data,test_cols)
    Remove_Unknown_Values(train_data,cols)
    Remove_Unknown_Values(test_data,test_cols)
    test_data = test_data.drop(columns='ID')
    train_labels = train_data['income>50K']
    train_data = train_data.drop(columns='income>50K')
    #We now have to remove the categorical attributes using a OneHotEncoding

    dec_tree = tree.DecisionTreeClassifier(criterion='entropy')
    encoder = prep.OneHotEncoder(handle_unknown='ignore')
    # encoded_train_data = encoder.fit_transform(train_data)
    # encoded_test_data = encoder.fit_transform(test_data)
    # train_tree = dec_tree.fit(encoded_train_data,train_labels)
    encoder.fit(train_data)
    encoded_train_data = encoder.transform(train_data).toarray()
    encoded_test_data = encoder.transform(test_data)

    # Perform_Decision_Tree(dec_tree,encoded_train_data,encoded_test_data,train_labels)
    Perform_AdaBoost(encoded_train_data,encoded_test_data,train_labels)

if __name__ == "__main__":
    main()
