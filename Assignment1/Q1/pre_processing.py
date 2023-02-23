"""
Author: Samuel Lovett (samuellovett@cmail.carleton.ca)
Maintainer: Samuel Lovett (samuellovett@cmail.carleton.ca)

This script handles the pre-processing of our training and test data.
acts as the main function which calls all the necessary functions required for question one. It starts by
calling the pre_processing script which conducts all the (feature analysis, feature visualization, feature cleaning,
and feature selection). pre_processing() returns the reduced and cleaned feature space which is then used as the input
for the classification methods. For simplicity the model evaluation is conducted directly within this script.

TODO:
    - separation of training and validation data sets
        - to stratify or not to stratify
    - feature analysis function
        - clarify what is meant
    - feature visualization function
        - visually examine for outliers
        - clarify what is meant
    - feature cleaning function
        - to standardize or not
        - to normalize or not
        - look for outliers
        - look for lines with missing data
            - potentially try to impute the missing data
    - feature selection function
    - return selected features along with labels for test, train, and validation

"""
import pandas as pd


class pre_processing:
    def __init__(self):
        print("hello")
        self.test_data = pd.read_csv('test.csv')
        self.whole_train_data = pd.read_csv('train.csv')
        # might not want to do these in the init because we likely want to do all of our cleaning before separating
        # Investigation required
        self.training_data = None
        self.validation_data = None

    def create_training_and_validation_set(self, stratified):
        if stratified:
            print("do a thing")
        else:
            print("do the other thing")
        self.training_data = "a subset of the whole_train_data with validation data removed"
        self.validation_data = "a subset of the whole_train_data with training data removed"
