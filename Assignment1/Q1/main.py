"""
Author: Samuel Lovett (samuellovett@cmail.carleton.ca)
Maintainer: Samuel Lovett (samuellovett@cmail.carleton.ca)

This script acts as the main function which calls all the necessary functions required for question one. It starts by
calling the pre_processing script which conducts all the (feature analysis, feature visualization, feature cleaning,
and feature selection). pre_processing() returns the reduced and cleaned feature space which is then used as the input
for the classification methods. For simplicity the model evaluation is conducted directly within this script.

TODO:
    - pre_processing script
    - decision tree script
    - knn script
    - Naive bayes script
    - SVM script
    - multi-layer perceptron script
    - model evaluation methods
    - report discussing results

"""
from pre_processing import pre_processing
import pandas as pd
import numpy


def read_test_data():
    # Load test data from CSV file
    test_data = pd.read_csv('test.csv', skiprows=1, header=0)
    # Split data into features (test_features) and class labels (test_labels)
    test_features = test_data.iloc[:, 1:-1].values
    return test_features


def read_train_data():
    # Load train data from CSV file
    col_names = ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome',
                 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'label']
    train_data = pd.read_csv('train.csv', skiprows=1, header=0, names=col_names)
    print(train_data.head())
    return train_data


def main():
    # import data
    test_data = read_test_data()
    train_features = read_train_data()[0]
    train_labels = read_train_data()[1]





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


