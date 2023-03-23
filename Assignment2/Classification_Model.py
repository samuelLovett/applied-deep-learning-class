import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold, cross_validate, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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
        print(best_mlp_params)
        self.mlp_classifier.set_params(**best_mlp_params)

        random_forest_grid_search = GridSearchCV(self.random_forest_classifier, random_forest_params, cv=5, n_jobs=-1)
        random_forest_grid_search.fit(self.val_features, self.val_labels)
        random_forest_best_params = random_forest_grid_search.best_params_
        print(random_forest_best_params)
        self.random_forest_classifier = RandomForestClassifier(**random_forest_best_params)

    def nested_cross_val_ensemble_test(self, the_features, the_labels):
        ensemble_clf = VotingClassifier(
            estimators=[('rf', self.random_forest_classifier), ('mlp', self.mlp_classifier)], voting='soft')

        my_scoring = ['accuracy', 'f1', 'precision', 'recall']
        outer_cv = KFold(n_splits=5)
        inner_cv = KFold(n_splits=5)

        # Compute accuracy using nested cross-validation
        outer_scores = []
        outer_precision = []
        outer_recall = []
        outer_f1 = []
        outer_roc_auc = []
        for train_index, test_index in outer_cv.split(the_features):
            X_train, X_test = the_features[train_index], the_features[test_index]
            y_train, y_test = the_labels[train_index], the_labels[test_index]

            best_score = 0
            best_precision = 0
            best_recall = 0
            best_f1 = 0
            best_roc_auc = 0
            for train_index_inner, test_index_inner in inner_cv.split(X_train):
                X_train_inner, X_val = X_train[train_index_inner], X_train[test_index_inner]
                y_train_inner, y_val = y_train[train_index_inner], y_train[test_index_inner]

                # Fit the ensemble classifier on the inner training set
                ensemble_clf.fit(X_train_inner, y_train_inner)

                # Compute the score on the inner validation set
                score = ensemble_clf.score(X_val, y_val)

                y_pred = ensemble_clf.predict(X_val)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, ensemble_clf.predict_proba(X_test)[:, 1])

                if score > best_score:
                    best_score = score
                elif precision > best_precision:
                    best_precision = precision
                elif recall > best_recall:
                    best_recall = recall
                elif f1 > best_f1:
                    best_f1 = f1
                elif roc_auc > best_roc_auc:
                    best_roc_auc = roc_auc

            # Fit the ensemble classifier on the outer training set
            ensemble_clf.fit(X_train, y_train)

            # Compute the score on the outer test set
            score = ensemble_clf.score(X_test, y_test)
            y_pred = ensemble_clf.predict(X_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, ensemble_clf.predict_proba(X_test)[:, 1])

            outer_scores.append(score)
            outer_precision.append(precision)
            outer_recall.append(recall)
            outer_f1.append(f1)
            outer_roc_auc.append(roc_auc)



        # Compute the mean accuracy and standard deviation across all folds
        mean_score = np.mean(outer_scores)
        std_score = np.std(outer_scores)

        df = pd.DataFrame({'Accuracy': [outer_scores], 'Precision': [outer_precision], 'Recall': [outer_recall],
                           'F1-score': [outer_f1], 'AUC': [outer_roc_auc]})
        df.loc['mean'] = df.mean()
        df.loc['std'] = df.std()
        df.to_excel('nested_cv_results.xlsx', index=False, na_rep='NaN')
        print(df.to_markdown())


    def repeated_k_fold_ensemble_test(self, the_test_x, the_test_y):
        # This is just a rough code of how it should work does not include any repeated k fold
        ensemble_clf = VotingClassifier(estimators=[('rf', self.random_forest_classifier), ('mlp', self.mlp_classifier)], voting='soft')

        my_scoring = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']

        # Define Repeated K-fold cross-validation
        rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
        # Evaluate accuracy using Repeated K-fold cross-validation
        scores = cross_validate(ensemble_clf, the_test_x, the_test_y, cv=rkf, scoring=my_scoring)
        df = pd.DataFrame(scores)
        df = df[['test_accuracy', 'test_f1', 'test_precision', 'test_recall', 'test_roc_auc']]
        df.loc['mean'] = df.mean()
        df.loc['std'] = df.std()
        df.to_excel('cv_results.xlsx', index=False)
        print(df.to_markdown())

