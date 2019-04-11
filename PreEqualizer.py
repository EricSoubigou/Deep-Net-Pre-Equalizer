# For DL libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim

from Utils import *


class PreEqualizer(nn.Module):
    """ Neural net of the pre-equlizer
    :fc_n: fully connected layers
    """

    def __init__(self, symb_nb):
        super(Net, self).__init__()
        # Set the loss as the MSE loss function. (Not the Cross entropy)
        self.loss_function = nn.MSELoss
        # Number of symbols accepted in the entry of the neural network
        self.symb_nb = symb_nb
        #  Fully connected layers
        self.fc1 = nn.Linear(2 * self.symb_nb, 256)
        self.fc2 = nn.Linear(2 * self.symb_nb, 256)
        self.fc3 = nn.Linear(2 * self.symb_nb, 256)
        # Init trainable scalar
        self.alpha = torch.randn(1, 1)

    def forward(self, symbols):
        """ Perform the feedforward process
        :symbols: A 1D float array, containing the symbols to pre-equalize
        """
        # First, adapat the input to the Neural network this means that we
        # have to handle the case where there is the CP
        pre_equ_symbols = np.array(symbols.shape, dtype=complex)
        # Loop on the different chunk of the signal
        for i in range(len(symbols) / self.symb_nb):
            # Convert the complex array to a 2D real array
            formated_symb = from_complex_to_real(
                symbols[i * self.symb_nb: (i + 1) * self.symb_nb]
            )
            # Then feed the neural network with the adapted symbol
            out_1 = F.relu(self.fc1(formated_symb))
            out_2 = F.relu(self.fc2(out_1))
            out_3 = F.linear(self.fc3(out_2))
            # Convert the output to a complex vector.
            pre_equ_symbols[
            i * self.symb_nb: (i + 1) * self.symb_nb
            ] = from_real_to_complex(out_3)
        # Lastly we sum the complex input with the weighted output of the network
        return symbols + self.alpha * pre_equ_symbols

    def backpropagation(self, output_symb, targeted_symb):
        """ Use to perform the back propagation update
        :output_symb:
        :targeted_symb:
        """
        loss = self.loss_function(output_symb, targeted_symb)
        loss.backward()

    def feedback_update(self, x_hat):
        """ Perform a SGD to update the weights given the results of
        the viterbi decoder
        :x_hat: TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """
        x_hat = 1

    # Static Methods

    @staticmethod
    def train(pre_equalizer, received_symb, targeted_symb, nb_epochs, sgd_step=0.01):
        """ To train the NN pre-equalizer.
        :pre_equalizer: A Pre_Equalizer, the one which will be trained
        :received_symb: A 1D array, received symbols from the OFDM
        :targeted_symb: A 1D array of the targeted symbols before the
            demaping process
        TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """
        optimizer = optim.SGD(pre_equalizer.parameters(), lr=sgd_step)
        # Loop on the number of epcohs
        for epoch in range(nb_epochs):
            for i, mini_batch in enumerate(data, 0):
                # Zero the gradient buffers
                optimizer.zero_grad()
                # Perform the forward operation
                pre_eq_symbols = pre_equalizer.forward(received_symb)
                # Launch the backward function
                pre_equalizer.backpropagation(pre_eq_symbols, targeted_symb)
                # Perform update of the gradient
                optimizer.step()
