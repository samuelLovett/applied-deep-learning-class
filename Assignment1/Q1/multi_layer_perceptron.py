from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV


class mlpClass:

    def __init__(self):
        self.mlp_classifier = None

    def train(self, dt_training_features, dt_training_label):
        self.mlp_classifier = MLPClassifier(hidden_layer_sizes=(50, 50, 50), max_iter=1000, random_state=42)
        self.mlp_classifier.fit(dt_training_features, dt_training_label)

    def tune_hyper_parameters(self, dt_validation_features, dt_validation_label):
        parameters_to_optimize = {'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (50,), (100,)],
                                  'activation': ['tanh', 'relu', 'logistic'],
                                  'alpha': [0.0001, 0.001, 0.05],
                                  'learning_rate': ['constant', 'adaptive'], }
        optimizer = RandomizedSearchCV(self.mlp_classifier, parameters_to_optimize, n_iter=10, cv=5)
        optimizer.fit(dt_validation_features, dt_validation_label)
        self.mlp_classifier = optimizer.best_estimator_
        # print("Best parameters:", optimizer.best_params_)
        # print("Best score:", optimizer.best_score_)

    def predict(self, dt_test_features):
        return self.mlp_classifier.predict(dt_test_features)
