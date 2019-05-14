from torch.utils.data import Dataset

import torch
import numpy as np
from Utils import from_complex_to_real

class OFDMSamplesDataset(Dataset):
    """
    Class which handle the data set of samples
    Each data set is saved in a numpy tuple where the first element is a matrix of size (nb_OFDM_symbols, nb_carriers)
    containing the samples received after the channel. The second element is another matrix with the same shape
    (nb_OFDM_symbols, nb_carriers) which is the targeted samples. They are the original samples present before the
    emission in the channel.

    """

    def __init__(self, file_path, *args):
        """

        :param file_name: A string, the file path of the data set that need to be loaded
        """
        # TODO  Make sure to handle the case where the file path doesn't exist.
        assert (file_path is not None), "File path isn't defined"

        # Load the file
        data_set = np.load(file_path)
        self.samples = data_set[0]
        self.targeted_samples = data_set[1]

        nb_data_set = len(args)
        if nb_data_set != 0:
            for id_set in range(nb_data_set):
                data_set = np.load(args[id_set])
                self.samples = np.concatenate((self.samples, data_set[0]), axis=0)
                self.targeted_samples = np.concatenate((self.targeted_samples, data_set[1]), axis=0)

    def __len__(self):
        """
        Return the length of the data set. In this case, it will be the number of OFDM symbols
        :return: A positive integer, the number of OFDM symbols (number of rows of the data set)
        """
        return self.samples.shape[0]

    def __getitem__(self, index):
        """
        Return an OFDM symbol from the data set
        :param index: A positive integer, index of the OFDM symbol in the data set
        :return: A tuple of complex numpy.array
        """

        return (torch.from_numpy(from_complex_to_real(self.samples[index, :])).float(),
                torch.from_numpy(from_complex_to_real(self.targeted_samples[index, :])).float())

    def get_number_of_carriers(self):
        """
        Give the number of carriers of the data set.
        :return: self.samples.shape[1]
        """
        return self.samples.shape[1]

    def get_dimensions(self):
        """
        Give shape of the data set
        :return: self.samples.shape
        """
        return self.samples.shape

    @staticmethod
    def save(samples, targeted_samples, file_path):
        """
        Save a data set with numpy in a tuple structure.
        :param file_path: A string, the filepath where the file will be located
        :param samples: A N,M-complex array containing the samples at the receiver
        :param targeted_samples: A N,M-complex array containing the targeted sample at the emiter.
        """
        #  Verify that the size of both matrix are consistent
        print("shape comparison samples and targets", samples.shape, targeted_samples.shape)
        assert (samples.shape == targeted_samples.shape), "Samples and targeted samples hasn't got the same shape : "

        data_set = (samples, targeted_samples)

        with open(file_path, "wb") as handle:
            np.save(handle, data_set)

        print("Data set created at " + file_path)

