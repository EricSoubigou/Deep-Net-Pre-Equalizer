# For DL libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim

import sys

from Utils import *


class PreEqualizer(nn.Module):
    """ Neural net of the pre-equlizer
    :fc_n: fully connected layers
    """

    def __init__(self, symb_nb):
        """
        Init the object of PreEqualizer
        :param symb_nb: A positive integer, number of symbols in an OFDM Symbols (ie. nb_carriers
            + cp_length + off_carriers)
        """
        super().__init__() #PreEqualizer, self
        # Set the loss as the MSE loss function. (Not the Cross entropy)
        self.loss_function = nn.MSELoss()
        # Number of symbols accepted in the entry of the neural network
        self.symb_nb = symb_nb
        #  Fully connected layers
        self.fc1 = nn.Linear(2 * self.symb_nb, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2 * self.symb_nb)
        # Init trainable scalar
        self.alpha = nn.Parameter(torch.randn(1, 1))
        self.tmp_frame = None

    def forward(self, symbols):
        """
        Perform the feedforward process
        :param symbols: A 1D float array, containing the symbols to pre-equalize
        """
        # We get through the neural net
        out_1 = F.relu(self.fc1(symbols))
        out_2 = F.relu(self.fc2(out_1))
        out_3 = self.fc3(out_2)
        # Multiply by a constant
        symbols = symbols + self.alpha * out_3
        return symbols

    def feedback_update(self, estimates, targets, sgd_step=0.001):
        """ Perform a SGD to update the weights given the results of
        the Viterbi decoder
        :param estimates:
        :param targets:
        :param sgd_step:
        """
        # Convert the data into readable
#estimates = torch.from_numpy(from_complex_to_real(estimates)).float()
        targets = torch.from_numpy(from_complex_to_real(targets)).float()
        # Zero the gradient buffers
        self.internal_optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.internal_optimizer.zero_grad()
        # Perform the pre-equalization.
        pre_eq_out = self(self.tmp_frame)
        # Compute the loss
        loss = self.loss_function(pre_eq_out, targets)
        #loss = self.loss_function(estimates, targets)
        # Perform the backward function
        loss.backward()
        # Perform update of the gradient
        self.internal_optimizer.step()


    # Static Methods
    @staticmethod
    def train(pre_equalizer, training_data_loader, validation_data_loader, nb_epochs, path_training, sgd_step=0.001):
        """ To train the NN pre-equalizer.
        :param pre_equalizer: A Pre_Equalizer, the one which will be trained
        :param data_loader: A DataLoader, create the mini batch for the training
        :param nb_epochs: A positive integer, number of epochs that will performed
        during the training process
        """
        # Loss vector init
        val_loss, train_loss = np.zeros((nb_epochs,1)), np.zeros((nb_epochs,1))
        # Init optimizer
        optimizer = optim.Adam(pre_equalizer.parameters(), lr=sgd_step)
        # Loop on the number of epochs
        for epoch in range(nb_epochs):
            # Loop on the training data set
            for batch_idx, (samples, targets) in enumerate(training_data_loader):
                # Zero the gradient buffers
                optimizer.zero_grad()
                # Perform the forward operation
                pre_eq_symbols = pre_equalizer.forward(samples)
                # Perform the backward function
                loss = pre_equalizer.loss_function(pre_eq_symbols, targets)
                loss.backward()
                # Perform update of the gradient
                optimizer.step()
            # Print loss
            sys.stdout.write("\rEpoch {}/{}; training MSE : {}".format(str(epoch + 1), str(nb_epochs), str(loss.item())))
            train_loss[epoch] = loss.item()

            # Launch on the validation data loader
            for batch_idx, (val_samples, val_targets) in enumerate(validation_data_loader):
                # Perform the forward operation
                pre_eq_symbols = pre_equalizer.forward(val_samples)
                # Launch the backward function
                validation_loss = pre_equalizer.loss_function(pre_eq_symbols, val_targets)
            # Print validation loss
            sys.stdout.write("; validation MSE : {}".format(str(validation_loss.item())))
            val_loss[epoch] = validation_loss.item()

        # Print the MSE in function of the epoch
        plt.plot(np.linspace(0, nb_epochs, nb_epochs), val_loss, "b", label="Validation loss")
        plt.plot(np.linspace(0, nb_epochs, nb_epochs), train_loss, "r", label="Training loss")
        #plt.yscale("log")
        plt.title("MSE loss performances")
        plt.xlabel("Epochs")
        plt.ylabel("MSE")
        plt.grid(True)
        plt.legend()
        plt.show()

        performances = {
            "val_loss" : val_loss,
            "train_loss" : train_loss,
        }

        # Save performances
        with open(path_training, "wb") as handle:
            pickle.dump(performances, handle)

        print("Performances of the training saved at " + path_training)


        #@staticmethod




