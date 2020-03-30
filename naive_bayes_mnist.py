import numpy as np 
import matplotlib.pyplot as plt 
from util import get_data
from datetime import datetime
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn 

class NaiveBayes(object):
    def fit(self, X, Y, smoothing=10e-3):
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y)
        for c in list(labels):
            current_x = X[Y == c]
            self.gaussians[c] = {
                'mean': current_x.mean(axis=0),
                'var': current_x.var(axis=0) + smoothing,
            }
            self.priors[c] = float(len(Y[Y == c])) / len(Y)
    
    def predict(self, X):
        K = len(self.gaussians)
        N = X.shape[0]
        prediction = np.zeros((N,K))
        for c,g in self.gaussians.items():
                mean , var = g['mean'] , g['var']
                prediction[:,c] = mvn.logpdf(X, mean=mean, cov=var) + np.log(self.priors[c])
        return np.argmax(prediction,axis =1)

    def score(self,X,Y):
        P = self.predict(X)
        return np.mean(P==Y)

   

if __name__ == "__main__":
    X , Y = get_data(1000)
    Ntrain = 700
    Xtrain , Ytrain = X[:Ntrain] , Y[:Ntrain]
    Xtest  , Ytest  = X[Ntrain:] , Y[Ntrain:]
    naive_bayes = NaiveBayes()
    naive_bayes.fit(Xtrain,Ytrain)
    print("train Accuracy = ",naive_bayes.score(Xtrain , Ytrain))
    print("test  Accuracy = ",naive_bayes.score(Xtest  , Ytest))