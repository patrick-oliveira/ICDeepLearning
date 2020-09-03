import numpy as np

class Perceptron(object):
    
    def __init__(self, eta=0.01, maxIter = 100):
        self.eta = eta
        self.maxIter = maxIter
    
    def adjust(self, X, y):
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis = 1)
        self.W = np.zeros(X.shape[1])
        self.Errors = [ np.sum((y - self.predict(X).reshape(len(y), 1))**2) ]
        index = np.arange(len(y))
        for i in range(self.maxIter):
            for Xi, Yi in zip(X[index], y[index]):
                update = self.eta * (Yi - self.predict(Xi))
                self.W += update*Xi
                
            self.Errors.append(np.sum((y - self.predict(X).reshape(len(y), 1))**2))
            if(self.Errors[-1] == 0):
                break
    
    def batchAdjust(self, X, y):
        n_values = X.shape[0]
        X = np.concatenate([np.ones((n_values, 1)), X], axis = 1)
        self.W = np.ones(X.shape[1]); self.W[0] = 0
        self.Errors = [1]
        i = 0
        
        while self.Errors[-1] != 0 and i < self.maxIter:
            missclassificationIndex = np.where((self.predict(X) - y.T )!= 0)[1]
            error = self.errorCorrection(X[missclassificationIndex], y[missclassificationIndex])
            self.Errors.append(error[0])
            self.W = self.W - self.eta * error[1]
            i += 1
            
        return self
    
    def predict(self, X):
        if(len(X.shape) > 1 and X.shape[1] != self.W.shape[0]):
            n_values = X.shape[0]
            X = np.concatenate([np.ones((n_values, 1)), X], axis = 1)
            
        product = np.dot(X, self.W)
        result = np.where(product > 0, 1, -1)
        return result
    
    def errorCorrection(self, X, y):
        return [-np.sum(np.dot(X, self.W)*y), -np.sum(X*y, axis = 0)]
        