import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=20, activation="tanh"):
        super().__init__()
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        - activation: Nonlinearity in cell; 'tanh' or 'relu'
        """
        #######################################################################
        # TODO: Build a simple one layer RNN with an activation with the      #
        # attributes defined above and a forward function below. Use the      #
        # nn.Linear() function as your linear layers.                         #
        # Initialse h as 0 if these values are not given.                     #
        #######################################################################
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x, h=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Optional hidden vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence
                 (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = []
        #######################################################################
        #                                YOUR CODE                            #
        #######################################################################

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return h_seq, h


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=20):
        super().__init__()
        #######################################################################
        # TODO: Build a one layer LSTM with an activation with the attributes #
        # defined above and a forward function below. Use the                 #
        # nn.Linear() function as your linear layers.                         #
        # Initialse h and c as 0 if these values are not given.               #
        #######################################################################

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x, h=None, c=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Hidden vector (nr_layers, batch_size, hidden_size)
        - c: Cell state vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence
                 (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        - c: Final cell state vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = None
        #######################################################################
        #                                YOUR CODE                            #
        #######################################################################

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return h_seq, (h, c)


class RNN_Classifier(torch.nn.Module):
    def __init__(self, classes=10, input_size=28, hidden_size=128,
                 activation="relu"):
        super(RNN_Classifier, self).__init__()
        #######################################################################
        #  TODO: Build a RNN classifier                                       #
        #######################################################################
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.classes = classes
        self.rnn = nn.RNN()
        
    def farward(self, x):
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

#    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
#        print('Saving model... %s' % path)
#        torch.save(self, path)


class LSTM_Classifier(torch.nn.Module):
    def __init__(self, classes=10, input_size=28, hidden_size=128):
        super(LSTM_Classifier, self).__init__()
        #######################################################################
        #  TODO: Build a LSTM classifier                                      #
        #######################################################################
        self.num_layers = 1
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.classes = classes
        
        self.lstm = nn.LSTM(input_size, hidden_size, self.num_layers)
        self.fc = nn.Linear(hidden_size, classes)
 
    def forward(self, x):
        """
        input size: torch.size([28, 32, 28])
        where 32 is the batch_size, and 28*28 is the size of pictures in mnist dataset.
        target size: torch.Size([32])
        we want to calculate the cross entropy between output and targets
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Forward propagate LSTM
        out, _ = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[-1, :, :])
        return out

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################

#    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
#        print('Saving model... %s' % path)
#        torch.save(self, path)
