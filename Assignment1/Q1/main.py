"""
Author: Samuel Lovett (samuellovett@cmail.carleton.ca)
Maintainer: Samuel Lovett (samuellovett@cmail.carleton.ca)

This script acts as the main function which calls all the necessary functions required for question one. It starts by
calling the pre_processing script which conducts all the (feature analysis, feature visualization, feature cleaning,
and feature selection). pre_processing() returns the reduced and cleaned feature space which is then used as the input
for the classification methods. For simplicity the model evaluation is conducted directly within this script.

TODO:
    - pre_processing script
    - decision tree script
    - knn script
    - Naive bayes script
    - SVM script
    - multi-layer perceptron script
    - model evaluation methods
    - report discussing results

"""
import pre_processing


def main():
    # import data
    my_pre_processing_object = pre_processing()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


