import pandas as pd


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
        print('test')