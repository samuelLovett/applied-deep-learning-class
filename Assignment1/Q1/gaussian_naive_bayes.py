from sklearn.naive_bayes import GaussianNB


class naiveBayesClass:
    def __init__(self):
        self.naive_bayes = None

    def train(self, dt_training_features, dt_training_label):
        self.naive_bayes = GaussianNB()
        self.naive_bayes.fit(dt_training_features, dt_training_label)

    def predict(self, dt_test_features):
        return self.naive_bayes.predict(dt_test_features)
