import numpy as np 
from sortedcontainers import SortedList

from util import get_data,get_donut
from datetime import datetime

class KNN(object):
    def __init__(self,k):
        self.k = k

    def fit(self,X,y):
        self.X = X
        self.y = y

    def predict(self,X):
        y = np.zeros(len(X))
        for i,x in enumerate(X): # X - basically inputs we want to predict(test_data) while self.X - training data
            sl = SortedList()
            for j, xl in enumerate(self.X):
                diff = x - xl
                dist = diff.dot(diff)
                if len(sl) < self.k :
                    sl.add((dist , self.y[j]))
                else:
                    max_elem = sl[-1]
                    max_dist = max_elem[0]
                    if  dist < max_dist:
                        del  sl[-1]
                        sl.add((dist , self.y[j]))
            votes = {}
            for dist , clz in sl:
                if votes == {} or clz not in list(votes.keys()):
                    votes[clz] = 1
                else:
                    votes[clz] += 1
            max_voter = np.argmax(np.array(list(votes.values())))
            y[i] = list(votes.keys())[max_voter]
        return y

    def score(self,X,y):
        p = self.predict(X)
        return np.mean(p == y)
            
if __name__ == "__main__":
    X , y = get_data(2000)
    Ntrain = 1000
    Xtrain , Ytrain = X[:Ntrain] , y[:Ntrain]
    Xtest  , Ytest  = X[Ntrain:] , y[Ntrain:]
    for k in [1,2,3,4,5]:
        print("################# k = {} ############".format(k))
        knn = KNN(k)
        t0 = datetime.now()
        knn.fit(Xtrain, Ytrain)
        print("training_accuracy = {}".format(knn.score(Xtrain, Ytrain)))
        print("test_accuracy = {}".format(knn.score(Xtest, Ytest)))