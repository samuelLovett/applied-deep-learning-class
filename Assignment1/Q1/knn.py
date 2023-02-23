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


class knnClass:
    def __init__(knn, x, y, k):
        knn.prediction_time_start = None
        knn.prediction_time_end = None
        knn.k = k
        training_features = x
        training_labels = y
        # Train k-nearest neighbors classifier
        knn.training_start_time = time.time()
        knn.clf = KNeighborsClassifier(n_neighbors=k)
        knn.clf.fit(training_features, training_labels)
        knn.training_end_time = time.time()

    def predict(knn, test_features):
        # Make predictions on test set
        knn.prediction_time_start = time.time()
        prediction = knn.clf.predict(test_features)
        knn.prediction_time_end = time.time()
        return prediction

    def time_stats_training(knn):
        # Calculate the amount of time it took to train the classifier, as well as classify the test data in seconds
        elapsed_training_time_seconds = knn.training_end_time - knn.training_start_time
        elapsed_training_time_milli = elapsed_training_time_seconds*1000
        # print(f"Time elapsed to train KNN classifier with k={knn.k}: {elapsed_training_time_milli} ms")
        return elapsed_training_time_milli

    def time_stats_classifying(knn):
        # Calculate the amount of time it took to classify the test data in seconds and then converted to milliseconds
        try:
            elapsed_classification_time_seconds = knn.prediction_time_end - knn.prediction_time_start
            elapsed_classification_time_milli = elapsed_classification_time_seconds * 1000
            # print(f"Time elapsed to classify using knn classifier with k={knn.k}: {elapsed_classification_time_milli} ms")
        except:
            print("Classification Time Elapsed Variables have likely not been defined")
        return elapsed_classification_time_milli