from pre_processing import SamsPreProcessingClass as prepro


def main():
    my_preprocessingObj = prepro(stats=True)
    oversampled_data = []
    oversampled_data = my_preprocessingObj.over_sample()
    train_features_over = oversampled_data[0]
    train_label_over = oversampled_data[1]
    my_test_features = my_preprocessingObj.test_features
    my_test_labels = my_preprocessingObj.test_labels

    my_preprocessingObj.feature_selection()



if __name__ == '__main__':
    main()



