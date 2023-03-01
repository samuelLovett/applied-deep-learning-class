from sklearn import svm
from sklearn.model_selection import GridSearchCV


class svmClass:

    def __init__(self):
        self.svm_classifier = None

    def train(self, dt_training_features, dt_training_label):
        self.svm_classifier = svm.SVC(kernel='linear', C=1.0)
        self.svm_classifier.fit(dt_training_features, dt_training_label)
        return self.svm_classifier

    def tune_hyper_parameters(self, dt_validation_features, dt_validation_label):
        parameters_to_optimize = {'C': [0.1, 1, 10],
                                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
        optimizer = GridSearchCV(self.svm_classifier, param_grid=parameters_to_optimize, cv=5)
        optimizer.fit(dt_validation_features, dt_validation_label)
        self.svm_classifier = optimizer.best_estimator_
        # print("Best parameters:", optimizer.best_params_)
        # print("Best score:", optimizer.best_score_)

    def predict(self, dt_test_features):
        return self.svm_classifier.predict(dt_test_features)


