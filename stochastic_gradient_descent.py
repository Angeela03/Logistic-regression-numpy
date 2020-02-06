# This is the code for stochastic gradient descent

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

lam = 0.01
learning_rate = 0.00005
epoch = 100000


class Logistic_Regression():
    def __init__(self, theta):
        self.theta = theta

    def sigmoid_function(self, xx):
        return (1 / (1 + np.exp(-xx)))

    def h_t(self, x):
        return np.array(self.sigmoid_function(np.dot(x.T, self.theta)))

    def cost_function(self, x, y):
        # This function calculates the cost function
        m = 1
        h = self.h_t(x)
        func = ((np.dot(-y.T, np.log(h)) - (np.dot((1 - y.T), np.log(1 - h))))) + (
                (lam / (2 * m)) * np.dot(self.theta, self.theta.T))
        return func

    def derivative(self, x, y):
        # This function calculates the derivative
        h = self.h_t(x)
        d = (np.dot(x, h - y) + (lam * self.theta))
        return d

    def fit(self, x, y):
        # For fitting the data
        # Maintain a list of epochs and loss function
        list_epoch = []
        list_loss = []
        for i in range(epoch):
            print(i)
            list_loss_stochastic = []
            # Since it is stochastic gradient descent we iterate over each samples
            for j in range(len(x)):
                x_j = x[j]
                y_j = y[j]
                self.theta = self.theta - learning_rate * self.derivative(x_j, y_j)
                loss_stochastic = self.cost_function(x_j, y_j)
                list_loss_stochastic.append(loss_stochastic)
            loss = sum(list_loss_stochastic) / len(list_loss_stochastic)
            list_epoch.append(i)
            list_loss.append(loss)
        print("Loss:", loss)
        return list_epoch, list_loss, self.theta

    def predict_prob(self, x1):
        # Predict probabilities after fitting
        list_pre = []
        for i in x1:
            pre = self.sigmoid_function(np.dot(i.T, self.theta))
            list_pre.append(pre)

        return list_pre

    def predict(self, x1):
        # Making predictions based on threshold
        pre = [1 if x >= 0.5 else 0 for x in self.predict_prob(x1)]
        return np.array(pre).reshape(-1, 1)


def main(x,y ):
    # Define theta
    theta = np.zeros((x.shape[1]))

    # Call the regression class and fit it
    lg = Logistic_Regression(theta)
    list_epoch, list_loss, theta = lg.fit(x, y)

    # For plotting the regression line
    x1 = np.linspace(min(x[:, 1]), max(x[:, 1]), 10)
    x2 = (0.5 - theta[0] - theta[1] * x1) / theta[2]
    plt.figure(figsize=(20, 20))
    y_one_dim = y.reshape(-1)
    plt.scatter(x[y_one_dim == 0][:, 1], x[y_one_dim == 0][:, 2], color='r', label="Class 0")
    plt.scatter(x[y_one_dim == 1][:, 1], x[y_one_dim == 1][:, 2], color='g', label='Class 1')
    plt.legend()
    plt.plot(x1, x2)
    plt.show()

    # For plotting the loss function against each epoch
    plt.plot(list_epoch, list_loss)
    plt.show()


