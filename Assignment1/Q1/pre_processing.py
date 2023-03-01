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
import numpy as np
import matplotlib.pyplot as plotter
import sklearn.preprocessing as pre_pro
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA


class pre_ProcessingClass:
    def __init__(self, unfiltered_data, training_data=True):
        # remove all rows with an empty cell of data
        self.filtered_data = unfiltered_data.dropna()

        if training_data:
            # Create a ColumnTransformer to apply OneHotEncoder to categorical columns only
            transformer = ColumnTransformer(
                transformers=[
                    ('onehot', pre_pro.OneHotEncoder(), [0, 1, 2, 3, 4, 10])  # encode column 0 only
                ], remainder='passthrough'  # pass through the numerical columns and data labels as is
            )
            filtered_data_encoded = transformer.fit_transform(self.filtered_data)
            self.filtered_encoded_data_frame = pd.DataFrame(filtered_data_encoded)
            self.filtered_encoded_data_frame.columns = ['Female', 'Male', 'Not Married', 'Married', 'Zero kids',
                                                        'One Kid',
                                                        'Two Kids', 'Three or more kids', 'Graduate', 'Not Graduate',
                                                        'Not Self Emp.', 'Self Emp.', 'Rural', 'Semiurban', 'Urban',
                                                        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                                                        'Loan_Amount_Term', 'Credit_History', 'Label']
            # Split the feature vectors and labels
            whole_training_features = self.filtered_encoded_data_frame.iloc[:, :-1].values
            whole_training_labels = self.filtered_encoded_data_frame.iloc[:, -1].values

            self.whole_training_features_dataframe = pd.DataFrame(whole_training_features)
            self.whole_training_features_dataframe.columns = ['Female', 'Male', 'Not Married', 'Married', 'Zero kids',
                                                              'One Kid',
                                                              'Two Kids', 'Three or more kids', 'Graduate',
                                                              'Not Graduate',
                                                              'Not Self Emp.', 'Self Emp.', 'Rural', 'Semiurban',
                                                              'Urban',
                                                              'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                                                              'Loan_Amount_Term', 'Credit_History']

            self.whole_training_labels_dataframe = pd.DataFrame(whole_training_labels)
            self.whole_training_labels_dataframe.columns = ['labels']

            # Split the data into training and validation/test set data
            self.train_features, remaining_features, self.train_labels, remaining_labels = \
                train_test_split(whole_training_features, whole_training_labels, test_size=0.5,
                                 random_state=42)

            # Split the remaining data into validation and test(metrics) set data
            self.metric_features, self.validation_features, self.metrics_labels, self.validation_labels = \
                train_test_split(remaining_features, remaining_labels, test_size=0.2,
                                 random_state=42)

        else:
            # Create a ColumnTransformer to apply OneHotEncoder to categorical columns only
            transformer = ColumnTransformer(
                transformers=[
                    ('onehot', pre_pro.OneHotEncoder(), [0, 1, 2, 3, 4, 10])  # encode column 0 only
                ], remainder='passthrough'  # pass through the numerical columns and data labels as is
            )
            test_features_encoded = transformer.fit_transform(self.filtered_data)
            encoded_test_data_frame = pd.DataFrame(test_features_encoded)
            encoded_test_data_frame.columns = ['Female', 'Male', 'Not Married', 'Married', 'Zero kids', 'One Kid',
                                               'Two Kids',
                                               'Three or more kids', 'Graduate', 'Not Graduate', 'Not Self Emp.',
                                               'Self Emp.',
                                               'Rural', 'Semiurban', 'Urban', 'ApplicantIncome',
                                               'CoapplicantIncome',
                                               'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

    def get_data_clean(self):
        return self.filtered_data

    def get_whole_training_features(self):
        return self.whole_training_features_dataframe

    def get_whole_training_labels(self):
        return self.whole_training_labels_dataframe

    def get_training_features(self):
        return self.train_features

    def get_training_labels(self):
        return self.train_labels

    def get_validation_features(self):
        return self.validation_features

    def get_validation_labels(self):
        return self.validation_labels

    def get_metric_features(self):
        return self.metric_features

    def get_metric_labels(self):
        return self.metric_labels

    def do_pca(self, components, data):
        standardized_data = pre_pro.scale(data)
        pca = PCA(n_components=components)
        pca.fit(standardized_data)
        principle_comp = pca.transform(standardized_data)
        return principle_comp

    def visualize_features_heatmap(self):
        transformer = ColumnTransformer(
            transformers=[
                ('onehot', pre_pro.OneHotEncoder(), [20])  # encode column 0 only
            ], remainder='passthrough'  # pass through the numerical columns and data labels as is
        )
        data_with_encoded_labels = transformer.fit_transform(self.filtered_encoded_data_frame)
        data_with_encoded_labels_data_frame = pd.DataFrame(data_with_encoded_labels)
        data_with_encoded_labels_data_frame.columns = ['Not Accepted', 'Accepted', 'Female', 'Male', 'Not Married',
                                                       'Married', 'Zero kids',
                                                       'One Kid',
                                                       'Two Kids', 'Three or more kids', 'Graduate', 'Not Graduate',
                                                       'Not Self Emp.', 'Self Emp.', 'Rural', 'Semiurban', 'Urban',
                                                       'ApplicantIncome', 'CoapplicantIncome',
                                                       'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
        reindexed_columns = ['Female', 'Male', 'Not Married', 'Married', 'Zero kids',
                             'One Kid', 'Two Kids', 'Three or more kids', 'Graduate', 'Not Graduate',
                             'Not Self Emp.', 'Self Emp.', 'Rural', 'Semiurban', 'Urban',
                             'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                             'Loan_Amount_Term', 'Credit_History', 'Not Accepted', 'Accepted']
        reindexed_data_frame = data_with_encoded_labels_data_frame.reindex(columns=reindexed_columns)
        correlation_coef_mat = np.corrcoef(reindexed_data_frame.astype(float), rowvar=False)
        fig, ax = plotter.subplots(figsize=(10, 8))
        plotter.imshow(correlation_coef_mat, cmap='hot', interpolation='nearest')
        plotter.colorbar()
        plotter.xticks(range(len(reindexed_data_frame.columns)), reindexed_data_frame.columns,
                       rotation=90)
        plotter.yticks(range(len(reindexed_data_frame.columns)), reindexed_data_frame.columns)
        plotter.title("Feature Correlation Heat Map")
        plotter.subplots_adjust(left=0.015, bottom=0.2, right=0.985, top=0.956)
        plotter.show()

    # def visualize_features_boxplot(self): #not useful since I already encoded my data using One Hot
    #     data = self.whole_training_features_dataframe
    #     column_names = data.columns
    #     data = data.apply(pd.to_numeric)
    #
    #     nplots = 4
    #     nfigs = int(np.ceil(len(column_names) / nplots))
    #     fig, axes = plotter.subplots(nfigs, 1, figsize=(10, 5 * nfigs))
    #     fig.subplots_adjust(hspace=0.5)
    #
    #     # iterate over subplots and create box plots
    #     for i, ax in enumerate(axes.flat):
    #         start = i * nplots
    #         end = start + nplots
    #         cols_subset = column_names[start:end]
    #         ax.boxplot(data[cols_subset].values)
    #         ax.set_xticklabels(cols_subset, rotation=45)
    #         ax.set_title(f"Box Plots of {', '.join(cols_subset)}")
    #
    #     plotter.show()
    #
    #     # fig, ax = plotter.subplots(figsize=(8, 6))
    #     # data.boxplot(ax=ax)
    #     # plotter.title("Box Plot of Features")
    #     # plotter.ylabel("Value")
    #     # plotter.show()
