import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif


class SamsPreProcessingClass:
    def __init__(self, stats=True):
        self.x_oversampled = None
        self.y_oversampled = None
        class_labels = pd.read_csv('Y.csv')
        features = pd.read_csv('X_CT.csv')
        if stats:
            num_class0 = (class_labels['Class'] == 0).sum()
            num_class1 = (class_labels['Class'] == 1).sum()

            print("Class 0 Items: {}".format(num_class0))
            print("Class 1 Items: {}".format(num_class1))
            print("The size of the dataset is: {}".format(len(features)))
            print("The data set has {} features".format(len(features.columns)))

        # Split the data into test and train when the pre_processing object is created
        self.train_feature, self.test_features, self.train_labels, self.test_labels = train_test_split(features,
                                                                                                       class_labels,
                                                                                                       test_size=0.2,
                                                                                                       random_state=1)

    def over_sample(self):
        oversampled = RandomOverSampler(sampling_strategy='minority')
        # double check that the data set is actually being combined correctly since they are from two separate files - high Sam

        self.x_oversampled, self.y_oversampled = oversampled.fit_resample(self.train_feature, self.train_labels)
        return self.x_oversampled, self.y_oversampled

    def feature_selection(self):
        selector = SelectKBest(score_func=f_classif, k=30)
        training_new = selector.fit_transform(self.x_oversampled, self.y_oversampled)
        print(training_new.head)
