from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


class decisionTreeClass:

    def __init__(self):
        self.tree_classifier = None

    def train(self, dt_training_features, dt_training_label):
        self.tree_classifier = DecisionTreeClassifier()
        self.tree_classifier.fit(dt_training_features, dt_training_label)

    def tune_hyper_parameters(self, dt_validation_features, dt_validation_label):
        parameters_to_optimize = {'max_depth': [2, 4, 6, 8],
                                  'min_samples_split': [2, 4, 6, 8],
                                  'min_samples_leaf': [1, 2, 3, 4]}
        optimizer = GridSearchCV(self.tree_classifier, param_grid=parameters_to_optimize, cv=5)
        optimizer.fit(dt_validation_features, dt_validation_label)
        self.tree_classifier = optimizer.best_estimator_
        print("Best parameters:", optimizer.best_params_)
        print("Best score:", optimizer.best_score_)

    def predict(self, dt_test_features):
        return self.tree_classifier.predict(dt_test_features)


