import numpy as np 
import matplotlib.pyplot as plt 
from util import get_data
from datetime import datetime
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn 

class Bayes(object):
    def fit(self, X, Y, smoothing=10e-3):
        D = X.shape[1]
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y)
        for c in list(labels):
            current_x = X[Y == c]
            self.gaussians[c] = {
                'mean': current_x.mean(axis=0),
                'cov': np.cov(current_x.T) + np.eye(D) * smoothing,
                }
            self.priors[c] = float(len(Y[Y == c])) / len(Y)
    
    def predict(self, X):
        K = len(self.gaussians)
        N = X.shape[0]
        prediction = np.zeros((N,K))
        for c,g in self.gaussians.items():
                mean , cov = g['mean'] , g['cov']
                prediction[:,c] = mvn.logpdf(X, mean=mean, cov=cov) + np.log(self.priors[c])
        return np.argmax(prediction,axis =1)


    def score(self,X,Y):
        P = self.predict(X)
        return np.mean(P==Y)

   

if __name__ == "__main__":
    X , Y = get_data(10000)
    Ntrain = 7000
    Xtrain , Ytrain = X[:Ntrain] , Y[:Ntrain]
    Xtest  , Ytest  = X[Ntrain:] , Y[Ntrain:]
    bayes = Bayes()
    bayes.fit(Xtrain,Ytrain)
    print("train Accuracy = ",bayes.score(Xtrain , Ytrain))
    print("test  Accuracy = ",bayes.score(Xtest  , Ytest))