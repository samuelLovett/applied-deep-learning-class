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
from sklearn.model_selection import train_test_split


class pre_processing:
    def __init__(self, unfiltered_data):
        # remove all rows with empty data
        filtered_data = unfiltered_data.dropna()
        # Split the data into X (feature vectors) and y (labels)
        whole_training_features = filtered_data.iloc[:, :-1].values
        whole_training_labels = filtered_data.iloc[:, -1].values
        self.train_features, self.validation_features, self.train_labels, self.validation_labels = \
            train_test_split(whole_training_features, whole_training_labels, test_size=0.2, random_state=42)
        return filtered_data

    def create_training_and_validation_set(self, filtered_data):

        y.head();
        self.training_data = "a subset of the whole_train_data with validation data removed"
        self.validation_data = "a subset of the whole_train_data with training data removed"

