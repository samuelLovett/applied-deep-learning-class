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
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA



class pre_processing:
    def __init__(self, unfiltered_data, training_data=True):
        # remove all rows with empty data
        self.filtered_data = unfiltered_data.dropna()
        if training_data:
            # Split the data into X (feature vectors) and y (labels)
            whole_training_features = self.filtered_data.iloc[:, :-1].values
            whole_training_labels = self.filtered_data.iloc[:, -1].values

            # Create a ColumnTransformer to apply OneHotEncoder to categorical columns only
            transformer = ColumnTransformer(
                transformers=[
                    ('onehot', OneHotEncoder(), [0, 1, 2, 3, 4, 10])  # encode column 0 only
                ], remainder='passthrough'  # pass through the numerical column(s) as is
            )
            whole_training_features_encoded = transformer.fit_transform(whole_training_features)
            encoded_features = pd.DataFrame(whole_training_features_encoded)
            encoded_features.columns = ['Female', 'Male', 'Not Married', 'Married', 'Zero kids', 'One Kid', 'Two Kids',
                                        'Three or more kids', 'Graduate', 'Not Graduate', 'Not Self Emp.', 'Self Emp.',
                                        'Rural', 'Semiurban', 'Urban', 'ApplicantIncome', 'CoapplicantIncome',
                                        'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

            self.train_features, self.validation_features, self.train_labels, self.validation_labels = \
                train_test_split(encoded_features, whole_training_labels, test_size=0.2, random_state=42)

    def get_data_clean(self):
        return self.filtered_data

    def get_training_features(self):
        return self.train_features

    def get_validation_features(self):
        return self.validation_features

    def get_training_labels(self):
        return self.train_labels

    def get_validation_labels(self):
        return self.validation_labels

    def PCA(self, components, data):
        pca = PCA(n_components=components)
        pca.fit(data)
        principle_comp = pca.transform(data)

        return principle_comp
