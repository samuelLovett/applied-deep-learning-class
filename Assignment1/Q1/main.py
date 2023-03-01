"""
Author: Samuel Lovett (samuellovett@cmail.carleton.ca)
Maintainer: Samuel Lovett (samuellovett@cmail.carleton.ca)

This script acts as the main function which calls all the necessary functions required for question one. It starts by
calling the pre_processing script which conducts all the (feature analysis, feature visualization, feature cleaning,
and feature selection). pre_processing() returns the reduced and cleaned feature space which is then used as the input
for the classification methods. For simplicity the model evaluation is conducted directly within this script.

"""
from pre_processing import pre_ProcessingClass
from decision_tree import decisionTreeClass
from knn import knnClass
from gaussian_naive_bayes import naiveBayesClass
from support_vector_machines import svmClass
from multi_layer_perceptron import mlpClass
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report

import pandas as pd
import numpy as np

feature_visualization = False


def read_test_data():
    col_names = ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome',
                 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']
    test_data_in = pd.read_csv('test.csv', skiprows=1, header=0, names=col_names,
                               usecols=lambda column: column not in ['Loan_ID'])
    return test_data_in


def read_train_data():
    # Load train data from CSV file
    col_names = ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome',
                 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'label']
    train_data = pd.read_csv('train.csv', skiprows=1, header=0, names=col_names,
                             usecols=lambda column: column not in ['Loan_ID'])
    return train_data


def get_metrics(predicted_labels, true_labels):
    positive_label = 'Y'
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, pos_label=positive_label)
    recall = recall_score(true_labels, predicted_labels, pos_label=positive_label)
    f1 = f1_score(true_labels, predicted_labels, pos_label=positive_label)
    cm = confusion_matrix(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels)

    metrics = {'accuracy': accuracy,
               'precision': precision,
               'recall': recall,
               'f1': f1,
               'confusion_matrix': cm,
               'classification_report': report}

    return metrics


def main():
    # import data
    raw_test_data = read_test_data()
    raw_train_data = read_train_data()

    # preprocess the data - remove
    my_training_preprocessing = pre_ProcessingClass(raw_train_data, training_data=True)
    my_test_preprocessing = pre_ProcessingClass(raw_test_data, training_data=False)
    if feature_visualization:
        my_training_preprocessing.visualize_features_heatmap()

    full_set_of_training_features = my_training_preprocessing.get_whole_training_features()
    full_set_of_training_labels = my_training_preprocessing.get_whole_training_labels()

    training_features = my_training_preprocessing.get_training_features()
    training_labels = my_training_preprocessing.get_training_labels()

    validation_features = my_training_preprocessing.get_validation_features()
    validation_labels = my_training_preprocessing.get_validation_labels()

    metric_features = my_training_preprocessing.get_metric_features()
    metric_labels = my_training_preprocessing.get_metric_labels()

    test_data = my_test_preprocessing.get_test_data()

    pca_of_full_data = my_training_preprocessing.do_pca(5, full_set_of_training_features)
    pca_of_training_data = my_training_preprocessing.do_pca(5, training_features)
    pca_of_validation_data = my_training_preprocessing.do_pca(5, validation_features)
    pca_of_metric_data = my_training_preprocessing.do_pca(5, metric_features)
    pca_of_test_data = my_training_preprocessing.do_pca(5, test_data)

    # training, classification on the held out portion of the metrics data, and testing

    my_decision_tree = decisionTreeClass()
    my_decision_tree.train(training_features, training_labels)
    my_decision_tree.tune_hyper_parameters(validation_features, validation_labels)
    tree_predictions = my_decision_tree.predict(metric_features)
    tree_metrics = get_metrics(tree_predictions, metric_labels)
    print(f"Tree {tree_metrics['classification_report']}\n" )
    final_test_predictions = my_decision_tree.predict(test_data)
    final_test_predictions = pd.DataFrame(final_test_predictions)
    final_test_predictions.to_csv('Decision_Tree_Test_Predictions.csv', index=False)


    my_knn = knnClass()
    my_knn.train_and_tune_knn(training_features, training_labels)
    knn_predictions = my_knn.predict(metric_features)
    knn_metrics = get_metrics(knn_predictions, metric_labels)
    print(f"knn {knn_metrics['classification_report']}\n")
    final_test_predictions = my_knn.predict(test_data)
    final_test_predictions = pd.DataFrame(final_test_predictions)
    final_test_predictions.to_csv('KNN_Test_Predictions.csv', index=False)

    my_nb = naiveBayesClass()
    my_nb.train(training_features, training_labels)
    nb_predictions = my_nb.predict(metric_features)
    nb_metrics = get_metrics(nb_predictions, metric_labels)
    print(f"nb {nb_metrics['classification_report']}\n")
    final_test_predictions = my_nb.predict(test_data)
    final_test_predictions = pd.DataFrame(final_test_predictions)
    final_test_predictions.to_csv('Naive_Bayes_Test_Predictions.csv', index=False)

    my_svm = svmClass()
    my_svm.train(training_features, training_labels)
    my_svm.tune_hyper_parameters(validation_features, validation_labels)
    svm_predictions = my_svm.predict(metric_features)
    svm_metrics = get_metrics(svm_predictions, metric_labels)
    print(f"svm {svm_metrics['classification_report']}\n")
    final_test_predictions = my_svm.predict(test_data)
    final_test_predictions = pd.DataFrame(final_test_predictions)
    final_test_predictions.to_csv('SVM_Test_Predictions.csv', index=False)

    my_mlp = mlpClass()
    my_mlp.train(training_features, training_labels)
    my_mlp.tune_hyper_parameters(validation_features, validation_labels)
    mlp_predictions = my_mlp.predict(metric_features)
    mlp_metrics = get_metrics(mlp_predictions, metric_labels)
    print(f"mlp {mlp_metrics['classification_report']}\n")
    final_test_predictions = my_mlp.predict(test_data)
    final_test_predictions = pd.DataFrame(final_test_predictions)
    final_test_predictions.to_csv('MLP_Test_Predictions.csv', index=False)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
