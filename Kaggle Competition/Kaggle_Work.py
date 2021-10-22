import numpy as np
import pandas as pd
from sklearn import tree
import sklearn
import sklearn.preprocessing as prep

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

    test_cols
    #We now have to remove the categorical attributes using a OneHotEncoding

    dec_tree = tree.DecisionTreeClassifier()
    encoder = prep.OneHotEncoder()
    encoded_train_data = encoder.fit_transform(train_data)
    encoded_test_data = encoder.fit_transform(test_data)
    train_tree = dec_tree.fit(encoded_train_data,train_data['income>50K'])

    test_result = train_tree.predict(encoded_test_data)

    print("hello")

if __name__ == "__main__":
    main()
