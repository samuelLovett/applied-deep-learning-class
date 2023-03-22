from pre_processing import SamsPreProcessingClass as prepro
from Classification_Model import SamsClassifierModel as classifier


def main():
    my_preprocessingObj = prepro(stats=True)
    my_test_features = my_preprocessingObj.test_features
    my_test_labels = my_preprocessingObj.test_labels

    my_preprocessingObj.over_sample()
    my_preprocessingObj.feature_selection()
    my_10_training_features = my_preprocessingObj.recursive_feature_elimination_method()
    my_training_labels = my_preprocessingObj.y_oversampled

    my_classifier = classifier(my_10_training_features, my_training_labels)

    my_classifier.ensemble_learning_classifier_train()
    my_classifier.ensemble_learning_classifier_optimization()


if __name__ == '__main__':
    main()



