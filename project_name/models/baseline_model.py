from sklearn.linear_model import SGDClassifier
import numpy as np

class Baseline:
    def __init__(self, loss='hinge', max_iter=1000):
        self.model = SGDClassifier(loss=loss, max_iter=max_iter, tol=1e-3)
    
    def partial_fit(self, X, y, classes=None):
        """Train on a batch of data."""
        self.model.partial_fit(X, y, classes=classes) #error here
        return self
    
    def predict(self, X):
        return self.model.predict(X)