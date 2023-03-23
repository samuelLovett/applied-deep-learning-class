import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class SamsPreProcessingClass:
    def __init__(self, stats=True):
        self.flattened_y_over = None
        self.training_features_step_c = None
        self.training_step_b = None
        self.x_oversampled = None
        self.y_oversampled = None
        class_labels = pd.read_csv('Y.csv', index_col='id')
        features = pd.read_csv('X_CT.csv', index_col='id')
        if stats:
            num_class0 = (class_labels['Class'] == 0).sum()
            num_class1 = (class_labels['Class'] == 1).sum()

            print("Class 0 Items: {}".format(num_class0))
            print("Class 1 Items: {}".format(num_class1))
            print("The size of the dataset is: {}".format(len(features)))
            print("The data set has {} features".format(len(features.columns)))

        # Split the data into test and train when the pre_processing object is created
        self.train_feature, self.test_features, self.train_labels, self.test_labels_wide = train_test_split(features,
                                                                                                       class_labels,
                                                                                                       test_size=0.2,
                                                                                                       random_state=1)
        tobe_flattened_y_test = self.test_labels_wide[['Class']]
        np_tobe_flattened_test = tobe_flattened_y_test.to_numpy()
        self.test_labels = np_tobe_flattened_test.ravel()

    def over_sample(self):
        oversampled = RandomOverSampler(sampling_strategy='minority')
        self.x_oversampled, self.y_oversampled = oversampled.fit_resample(self.train_feature, self.train_labels)
        # print("Length of oversampled features {}".format(len(self.x_oversampled)))
        # print("Length of oversampled labels {}".format(len(self.y_oversampled)))
        # num_class0 = (self.y_oversampled['Class'] == 0).sum()
        # num_class1 = (self.y_oversampled['Class'] == 1).sum()
        #
        # print("Class 0 Items: {}".format(num_class0))
        # print("Class 1 Items: {}".format(num_class1))

    def feature_selection(self):
        # Set up the feature selector to select the 30 best features using f-score
        selector = SelectKBest(score_func=f_classif, k=30)
        tobe_flattened_y_over = self.y_oversampled[['Class']]
        np_tobe_flattened = tobe_flattened_y_over.to_numpy()
        self.flattened_y_over = np_tobe_flattened.ravel()
        # select the features
        selector.fit(self.x_oversampled, self.flattened_y_over)
        column_idx = selector.get_support(indices=True)
        self.training_step_b = self.x_oversampled.iloc[:,column_idx]

        # debug
        # num_rows, num_cols = self.training_step_b.shape
        # print("number of rows = {}".format(num_rows))
        # print("number of col = {}".format(num_cols))

    def recursive_feature_elimination_method(self):
        rand_forest_classif = RandomForestClassifier()
        rand_forest_elim = RFE(estimator=rand_forest_classif, n_features_to_select=10)
        rand_forest_elim.fit(self.training_step_b, self.flattened_y_over)

        selected_features = self.training_step_b.columns[rand_forest_elim.support_]

        self.training_features_step_c = self.training_step_b[selected_features]
        return self.training_features_step_c

