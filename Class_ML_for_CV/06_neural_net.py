import numpy as np 
import matplotlib.pyplot as plt

def lrelu(X):
	return np.maximum(X, 0.1*X)

def lrelu_derivative(X):
    lrelu_g = np.zeros_like(X)
    lrelu_g[X > 0] = 1.
    lrelu_g[X <= 0] = 0.1
    return lrelu_g

class FullyConnectedNet():
    def __init__(self, X, hidden_nodes, output_dim=2):

        input_dim = X.shape[1]

        # Xavier Initalization of weights
        # We use np.sqrt(1/N_avg) where N_avg = (N_{l-1}+N_{l})/2
        # to account for signal preservation in the backward pass as well
        self.W1 = np.random.randn(input_dim, hidden_nodes) * np.sqrt(2 / (input_dim + hidden_nodes))
        self.b1 = np.zeros((1, hidden_nodes))
        self.W2 = np.random.randn(hidden_nodes, hidden_nodes) * np.sqrt(2 / (2 * hidden_nodes))
        self.b2 = np.zeros((1, hidden_nodes))
        self.W3 = np.random.randn(hidden_nodes, output_dim) * np.sqrt(2 / (hidden_nodes + output_dim))
        self.b3 = np.zeros((1, output_dim))

    def feed_forward(self, x):
        h1 = lrelu(x.dot(self.W1) + self.b1)
        h2 = lrelu(h1.dot(self.W2) + self.b2)
        out = h2.dot(self.W3) + self.b3
        return h1, h2, out

    def mse_loss(self, y_pred, t):
        return 1/(2*y_pred.shape[0]) * np.sum((t - y_pred)**2)

    def backprop(self, X, y, h1, h2, output):
        dEdy = 1/X.shape[0] * (output-y)
        dW3 = (h2.T).dot(dEdy)
        db3 = np.sum(dEdy, axis=0, keepdims=True)
        
        dh2 = dEdy.dot(self.W3.T) * lrelu_derivative(h2)
        dW2 = (h1.T).dot(dh2)
        db2 = np.sum(dh2, axis=0, keepdims=True)

        dh1 = dh2.dot(self.W2.T) * lrelu_derivative(h1)
        dW1 = np.dot(X.T, dh1)
        db1 = np.sum(dh1, axis=0)

        return dW1, dW2, dW3, db1, db2, db3

    def train(self, X, y, num_passes=100000, learning_rate=0.1, batch_size=100):
        losses = []
        for i in range(num_passes):
            # SGD sample batch
            idcs = np.random.choice(np.arange(len(X)), batch_size, replace=False)
            # feed forward
            h1,h2,output = self.feed_forward(X[idcs])
            # backpropagation
            dW1, dW2, dW3, db1, db2, db3 = self.backprop(X[idcs],y[idcs], h1, h2, output)
            # update weights and biases
            self.W1 -= learning_rate * dW1
            self.W2 -= learning_rate * dW2
            self.W3 -= learning_rate * dW3
            self.b1 -= learning_rate * db1
            self.b2 -= learning_rate * db2
            self.b3 -= learning_rate * db3

            if i % 200 == 0:
                loss = self.mse_loss(y[idcs], output)
                print("Loss after iteration %i: %f" %(i, loss))  #uncomment once testing finished, return mod val to 1000

if __name__ == "__main__":

    learning_rate = 0.01 # learning rate for gradient descent
    batch_size = 20
    x_train = np.random.rand(100,1) * 4*np.pi
    y_train = np.sin(x_train)

    # Train network
    network = FullyConnectedNet(x_train, hidden_nodes=100, output_dim=1)
    network.train(x_train, y_train, batch_size=batch_size, learning_rate=learning_rate)

    # Test network
    x_test_1, x_test_2 = 2*np.pi * np.random.rand(100,1), -2*np.pi * np.random.rand(100,1)
    y_test_1, y_test_2 = np.sin(x_test_1), np.sin(x_test_2)

    pred = network.feed_forward(x_train)[-1]
    pred_1 = network.feed_forward(x_test_1)[-1]
    pred_2 = network.feed_forward(x_test_2)[-1]

    plt.scatter(x_train, y_train, label='Training data: {:03f} MSE'.format(network.mse_loss(pred, y_train)))
    plt.scatter(x_test_1, pred_1, label='Test data in domain: {:03f} MSE'.format(network.mse_loss(pred_1, y_test_1)))
    plt.scatter(x_test_2, pred_2, label='Test data out of domain: {:03f} MSE'.format(network.mse_loss(pred_2, y_test_2)))
    plt.legend()
    plt.show()