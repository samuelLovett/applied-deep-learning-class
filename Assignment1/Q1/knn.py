"""
Author: Samuel Lovett (samuellovett@cmail.carleton.ca)
Maintainer: Samuel Lovett (samuellovett@cmail.carleton.ca)

This script performs the knn model building. When predict is called it returns the predicted class labels based on the
classification parameters set previously.

TODO:
    - Optimization (where applicable)
    - training (DONE)
    - validation
    - hyper parameter tuning
    - prediction function (DONE)
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.model_selection import GridSearchCV
import numpy as np


class knnClass:
    def __init__(self):
        self.prediction_time_start = None
        self.prediction_time_end = None
        self.knn = None
        self.training_start_time = None
        self.training_end_time = None

    def train_and_tune_knn(self, x, y):
        training_features = x
        training_labels = y
        self.training_start_time = time.time()

        number_of_neighbours = {'n_neighbors': np.arange(1, 11)}
        self.knn = KNeighborsClassifier()
        optimize = GridSearchCV(self.knn, param_grid=number_of_neighbours, cv=5)
        optimize.fit(training_features, training_labels)
        self.knn = optimize.best_estimator_
        # print("Best parameters:", optimize.best_params_)
        # print("Best score:", optimize.best_score_)
        self.knn.fit(training_features, training_labels)
        self.training_end_time = time.time()

    def predict(self, knn_test_features):
        # Make predictions on test set
        self.prediction_time_start = time.time()
        prediction = self.knn.predict(knn_test_features)
        self.prediction_time_end = time.time()
        return prediction

    def time_stats_training(self):
        # Calculate the amount of time it took to train the classifier, as well as classify the test data in seconds
        elapsed_training_time_seconds = self.training_end_time - self.training_start_time
        elapsed_training_time_milli = elapsed_training_time_seconds*1000
        # print(f"Time elapsed to train KNN classifier with k={knn.k}: {elapsed_training_time_milli} ms")
        return elapsed_training_time_milli

    def time_stats_classifying(self):
        # Calculate the amount of time it took to classify the test data in seconds and then converted to milliseconds
        try:
            elapsed_classification_time_seconds = self.prediction_time_end - self.prediction_time_start
            elapsed_classification_time_milli = elapsed_classification_time_seconds * 1000
            # print(f"Time elapsed to classify using knn classifier with k={knn.k}: {elapsed_classification_time_milli} ms")
        except:
            print("Classification Time Elapsed Variables have likely not been defined")
        return elapsed_classification_time_milli