import numpy as np 
import matplotlib.pyplot as plt 
from util import get_data as mnist
from datetime import datetime

def get_data():
    X = np.random.random((300,2)) * 2 - 1
    w = np.array([-0.5,0.5])
    b = 0.1
    Y = np.sign(X.dot(w) + b)
    return X,Y

class Perceptron:
    def fit(self,X,Y, learning_rate = 1.0, epochs = 1000):
        N , D = X.shape
        self.W = np.random.randn(D)
        self.b = 0

        costs = []
        for epoch in range(epochs):
            y_pred = self.predict(X)
            incorrect_y = np.nonzero(y_pred != Y)[0]
            if len(incorrect_y) == 0 :
                print("Model got fully trained when {} epoch reached !!!".format(epoch))
                break
            random_pt = np.random.choice(incorrect_y)
            x , y = X[random_pt] , Y[random_pt]
            self.W += learning_rate*x*y
            self.b += learning_rate*y

            cost = len(incorrect_y)/N
            costs.append(cost)
        #print("Final W : {} and Final b : {}".format(self.W,self.b))
        plt.plot(costs)
        plt.show()

    def predict(self,X):
        # return np.sign(X.dot(self.W) + self.b)
        N = X.shape[0]
        y_pred = np.empty(N)
        for i in range(N):
            if X[i].dot(self.W) + self.b > 0:
                y_pred[i] = 1
            elif X[i].dot(self.W) + self.b < 0:
                y_pred[i] = -1
        return y_pred

    def score(self,X,Y):
        y_pred = self.predict(X)
        return np.mean(y_pred == Y)

    def get_mnist(self):
        X ,Y = mnist()
        binary_y = np.logical_or(Y == 0 , Y == 1)
        X = X[binary_y]
        Y = Y[binary_y]
        Y[Y == 0] = -1
        
        return X , Y
    
    def k_fold_cv(self,size,i,X,Y):
        Xval , Yval = X[(i-1)*size:i*size,:] , Y[(i-1)*size:i*size]
        Xtrain , Ytrain = np.concatenate((X[:(i-1)*size,:], X[(i+1)*size:,:]), axis=0), np.concatenate((Y[:(i-1)*size] , Y[(i+1)*size:]), axis=0)                                   
        return Xval , Yval , Xtrain , Ytrain 

if __name__ == "__main__":
    perceptron = Perceptron()
    X , Y = perceptron.get_mnist()

    k = 5
    size = len(Y) // k
    train_score = []
    test_score = []
    for i in range(1,k+1):
        Xval , Yval , Xtrain , Ytrain = perceptron.k_fold_cv(size,i,X,Y)
        perceptron.fit(Xtrain,Ytrain) 
        print("Fold {}".format(i)) 
        print("Train Accuracy :", perceptron.score(Xtrain , Ytrain))
        print("Test Accuracy :" , perceptron.score(Xval  , Yval))

        train_score.append(perceptron.score(Xtrain , Ytrain))
        test_score.append(perceptron.score(Xval  , Yval))

    print("Average Train Accuracy :", np.mean(train_score))
    print("Average Test Accuracy :" , np.mean(test_score))