import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.neural_network import MLPClassifier


class SamsClassifierModel:
    def __init__(self, whole_feature_data, whole_label_data):
        # split into validation and training
        self.mlp_classifier = None
        self.random_forest_classifier = None
        self.model_train_features, self.val_features, self.model_train_labels_wide, self.val_labels_wide = train_test_split(
            whole_feature_data,
            whole_label_data,
            test_size=0.2,
            random_state=1)
        tobe_flattened_y = self.model_train_labels_wide[['Class']]
        np_tobe_flattened = tobe_flattened_y.to_numpy()
        self.model_train_labels = np_tobe_flattened.ravel()

        tobe_flattened_y_val = self.val_labels_wide[['Class']]
        np_tobe_flattened_val = tobe_flattened_y_val.to_numpy()
        self.val_labels = np_tobe_flattened_val.ravel()


    def ensemble_learning_classifier_train(self):
        self.mlp_classifier = MLPClassifier(hidden_layer_sizes=(50, 50, 50), max_iter=500, random_state=42)
        self.mlp_classifier.fit(self.model_train_features, self.model_train_labels)

        self.random_forest_classifier = RandomForestClassifier()
        self.random_forest_classifier.fit(self.model_train_features, self.model_train_labels)


    def ensemble_learning_classifier_optimization(self):
        mlp_grid_param = {'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (50,), (100,)],
                                  'activation': ['tanh', 'relu', 'logistic'],
                                  'alpha': [0.0001, 0.001, 0.05],
                                  'learning_rate': ['constant', 'adaptive']}
        random_forest_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]}

        mlp_grid_search = GridSearchCV(estimator=self.mlp_classifier, param_grid=mlp_grid_param, cv=5, n_jobs=-1)
        mlp_grid_search.fit(self.val_features, self.val_labels)
        best_mlp_params = mlp_grid_search.best_params_
        self.mlp_classifier.set_params(**best_mlp_params)

        random_forest_grid_search = GridSearchCV(self.random_forest_classifier, random_forest_params, cv=5, n_jobs=-1)
        random_forest_grid_search.fit(self.val_features, self.val_labels)
        random_forest_best_params = random_forest_grid_search.best_params_
        self.random_forest_classifier = RandomForestClassifier(**random_forest_best_params)

    def ensemble_test(self, the_features, the_labels):
        print("lmao")
