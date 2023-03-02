"""
Author: Samuel Lovett (samuellovett@cmail.carleton.ca)
Maintainer: Samuel Lovett (samuellovett@cmail.carleton.ca)

This script acts as the main function which calls all the necessary functions required for question one. It starts by
calling the pre_processing script which conducts all the (feature analysis, feature visualization, feature cleaning,
and feature selection). pre_processing() returns the reduced and cleaned feature space which is then used as the input
for the classification methods. For simplicity the model evaluation is conducted directly within this script.

"""
from decision_tree import decisionTreeClass
from knn import knnClass
from gaussian_naive_bayes import naiveBayesClass
from support_vector_machines import svmClass
from multi_layer_perceptron import mlpClass
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from tensorflow.keras.datasets import mnist
from sklearn.ensemble import VotingClassifier
import numpy as np


def train_ensemble_classifier(dt_classifier, knn_classifier, nb_classifier, svm_classifier, mlp_classifier,
                              voting_type, features, labels):
    if voting_type == 'hard':
        ensemble_classifier = VotingClassifier(estimators=[('dt', dt_classifier), ('knn', knn_classifier),
                                                           ('gnb', nb_classifier), ('svc', svm_classifier),
                                                           ('mlp', mlp_classifier)], voting='hard')
        ensemble_classifier.fit(features, labels)
        return ensemble_classifier
    else:
        ensemble_classifier = VotingClassifier(estimators=[('dt', dt_classifier), ('knn', knn_classifier),
                                                           ('gnb', nb_classifier), ('svc', svm_classifier),
                                                           ('mlp', mlp_classifier)], voting='soft', n_jobs=-1)
        ensemble_classifier.fit(features, labels)
        return ensemble_classifier


def train_ensemble_classifier_test(dt_classifier, knn_classifier, nb_classifier, voting_type, features, labels):
    if voting_type == 'hard':
        ensemble_classifier = VotingClassifier(estimators=[('dt', dt_classifier), ('knn', knn_classifier),
                                                           ('gnb', nb_classifier)], voting='hard')
        ensemble_classifier.fit(features, labels)
        return ensemble_classifier
    else:
        ensemble_classifier = VotingClassifier(estimators=[('dt', dt_classifier), ('knn', knn_classifier),
                                                           ('gnb', nb_classifier)], voting='soft')
        ensemble_classifier.fit(features, labels)
        return ensemble_classifier


def get_metrics(predicted_labels, true_labels):
    report = classification_report(true_labels, predicted_labels)
    return report


def main():
    # import data
    (trainX, trainy), (test_features, test_labels) = mnist.load_data()
    training_features_3d = trainX[:50000, :, :]
    training_features = training_features_3d.reshape(training_features_3d.shape[0], -1)
    validation_features_3d = trainX[50000:, :, :]
    validation_features = validation_features_3d.reshape(validation_features_3d.shape[0], -1)
    training_labels = trainy[:50000]
    validation_labels = trainy[50000:]

    test_features_3d = test_features[:50000, :, :]
    test_data = test_features_3d.reshape(test_features_3d.shape[0], -1)
    test_labels=test_labels[:50000]

    # # preprocess the data - remove
    # my_training_preprocessing = pre_ProcessingClass(raw_train_data, training_data=True)
    # my_test_preprocessing = pre_ProcessingClass(raw_test_data, training_data=False)
    # if feature_visualization:
    #     my_training_preprocessing.visualize_features_heatmap()
    #
    # full_set_of_training_features = my_training_preprocessing.get_whole_training_features()
    # full_set_of_training_labels = my_training_preprocessing.get_whole_training_labels()
    #
    # training_features = my_training_preprocessing.get_training_features()
    # training_labels = my_training_preprocessing.get_training_labels()
    #
    # validation_features = my_training_preprocessing.get_validation_features()
    # validation_labels = my_training_preprocessing.get_validation_labels()
    #
    # metric_features = my_training_preprocessing.get_metric_features()
    # metric_labels = my_training_preprocessing.get_metric_labels()
    #
    # test_data = my_test_preprocessing.get_test_data()
    #
    # pca_of_full_data = my_training_preprocessing.do_pca(5, full_set_of_training_features)
    # pca_of_training_data = my_training_preprocessing.do_pca(5, training_features)
    # pca_of_validation_data = my_training_preprocessing.do_pca(5, validation_features)
    # pca_of_metric_data = my_training_preprocessing.do_pca(5, metric_features)
    # pca_of_test_data = my_training_preprocessing.do_pca(5, test_data)
    #
    # # training, classification on the held out portion of the metrics data, and testing

    my_decision_tree = decisionTreeClass()
    trained_decision_tree = my_decision_tree.train(training_features, training_labels)
    tree_predictions = my_decision_tree.predict(validation_features)
    tree_validation_metrics = get_metrics(tree_predictions, validation_labels)
    print(f"Tree Validation {tree_validation_metrics}\n")

    my_knn = knnClass()
    trained_knn = my_knn.train_and_tune_knn(training_features, training_labels)
    knn_predictions = my_knn.predict(validation_features)
    knn_validation_metrics = get_metrics(knn_predictions, validation_labels)
    print(f"knn {knn_validation_metrics}\n")

    my_nb = naiveBayesClass()
    trained_nb = my_nb.train(training_features, training_labels)
    nb_predictions = my_nb.predict(validation_features)
    nb_validation_metrics = get_metrics(nb_predictions, validation_labels)
    print(f"nb {nb_validation_metrics}\n")

    my_svm = svmClass()
    trained_svm = my_svm.train(training_features, training_labels)
    svm_predictions = my_svm.predict(validation_features)
    svm_validation_metrics = get_metrics(svm_predictions, validation_labels)
    print(f"svm {svm_validation_metrics}\n")

    my_mlp = mlpClass()
    trained_mlp = my_mlp.train(training_features, training_labels)
    mlp_predictions = my_mlp.predict(validation_features)
    mlp_validation_metrics = get_metrics(mlp_predictions, validation_labels)
    print(f"mlp {mlp_validation_metrics}\n")

    trained_ensemble_hard = train_ensemble_classifier(trained_decision_tree, trained_knn, trained_nb, trained_svm,
                                                      trained_mlp, 'hard', training_features, training_labels)
    ensemble_predict_hard = trained_ensemble_hard.predict(validation_features)
    ensemble_hard_validation_metrics = get_metrics(ensemble_predict_hard, validation_labels)
    print(f"Hard ensemble Validation {ensemble_hard_validation_metrics}\n")

    #
    trained_ensemble_soft = train_ensemble_classifier(trained_decision_tree, trained_knn, trained_nb, trained_svm,
                                                      trained_mlp, 'soft', training_features, training_labels)
    ensemble_predict_soft = trained_ensemble_soft.predict(validation_features)
    ensemble_soft_validation_metrics = get_metrics(ensemble_predict_soft, validation_labels)
    print(f"Soft ensemble Validation {ensemble_soft_validation_metrics}\n")

    ensemble_test_predict = trained_ensemble_hard.predict(test_data)
    ensemble_test_metrics = get_metrics(ensemble_test_predict, test_labels)
    print(f"Hard ensemble Test Metrics {ensemble_test_metrics}\n")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
