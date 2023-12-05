# This is the code for gradient descent

import numpy as np
from matplotlib import pyplot as plt

# Define parameters
lam = 0.0001
learning_rate = 0.005
epoch = 500000


class Logistic_Regression():
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def sigmoid_function(self, xx):
        return (1 / (1 + np.exp(-xx)))

    def h_t(self):
        return self.sigmoid_function(np.dot(self.x, self.theta))

    def cost_function(self):
        # This function calculates the cost function
        m = self.x.shape[0]
        h = self.h_t()
        func = (np.sum((-self.y * np.log(h)) - ((1 - self.y) * np.log(1 - h))) / m) + (
                (lam / (2 * m)) * np.sum(self.theta**2))
        return func

    def derivative(self):
        # This function calculates the derivative
        m = self.x.shape[0]
        h = self.h_t()
        d = (np.dot(self.x.T, h - self.y) + (lam * self.theta)) / m
        return d

    def fit(self):
        # For fitting the data
        # Maintain a list of epochs and loss function
        list_epoch = []
        list_loss = []
        for i in range(epoch):
            print(i)
            self.theta = self.theta - learning_rate * self.derivative()
            list_epoch.append(i)
            list_loss.append(self.cost_function())
        print("Loss:", self.cost_function())
        return list_epoch, list_loss, self.theta

    def predict_prob(self, x1):
        # Predict probabilities after fitting
        return self.sigmoid_function(np.dot(x1, self.theta))

    def predict(self, x1):
        # Making predictions based on threshold
        pre = [1 if x >= 0.5 else 0 for x in self.predict_prob(x1)]
        return np.array(pre).reshape(-1, 1)


def main(x, y):
    # x and y are train_x and train_y
    # Define theta
    theta = np.zeros((x.shape[1], 1))

    # Call the regression class and fit it
    lg = Logistic_Regression(x, y, theta)
    list_epoch, list_loss, theta = lg.fit()

    # For plotting the regression line
    x1 = np.linspace(min(x[:, 1]), max(x[:, 1]), 10)
    x2 = (0.5 - theta[0][0] - theta[1][0] * x1) / theta[2][0]
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

x=np.random.rand(10,5)
y=np.zeros(10)
main(x, y)
