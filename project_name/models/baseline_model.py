from sklearn.linear_model import SGDClassifier
import numpy as np

class Baseline:
    def __init__(self, loss='hinge', max_iter=1000):
        self.model = SGDClassifier(loss=loss, max_iter=max_iter, tol=1e-3)
    
    def partial_fit(self, X, y, classes=None):
        """Train on a batch of data."""
        X = X.view(X.size(0), -1).numpy()     # flatten and convert to NumPy
        y = y.numpy()                         # convert labels to NumPy
        if y.ndim > 1:                        # if one-hot encoded
            y = np.argmax(y, axis=1)         # convert to class indices
        self.model.partial_fit(X, y, classes=classes)
        return self
    
    def predict(self, X):
        X = X.view(X.size(0), -1).numpy()     # flatten input for prediction too
        return self.model.predict(X)