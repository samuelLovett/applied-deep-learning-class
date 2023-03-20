import pandas as pd
from imblearn.over_sampling import RandomOverSampler

class SamsPreProcessingClass:
    def __init__(self, stats=True):
        class_labels = pd.read_csv('Y.csv')
        features = pd.read_csv('X_CT.csv')
        if stats:
            num_class0 = (class_labels['Class'] == 0).sum()
            num_class1 = (class_labels['Class'] == 1).sum()

            print("Class 0 Items: {}".format(num_class0))
            print("Class 1 Items: {}".format(num_class1))
            print("The size of the dataset is: {}".format(len(features)))
            print("The data set has {} features".format(len(features.columns)))

    def over_sample(self):
        oversample = RandomOverSampler(sampling_strategy='minority')
        #double check that the data set is actually being combined correctly since they are from two seperate files - high Sam
        x_ct = pd.read_csv('X_CT.csv')
        y_ct = pd.read_csv('Y.csv')

        x_oversampled, y_oversampled = oversample.fit_resample(x_ct, y_ct)

        """
        After you validated the above use something like this
        steps = [('over', RandomOverSampler()), ('model', DecisionTreeClassifier())]
        pipeline = Pipeline(steps=steps)
        # evaluate pipeline
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        
        for why check https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/
        """


