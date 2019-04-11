from torch.utils.data import Dataset

import numpy as np

class OFDMSamplesDataset(Dataset):
    """
    Class which handle the data set of samples
    Each data set is saved in a numpy tuple where the first element is a matrix of size (nb_OFDM_symbols, nb_carriers)
    containing the samples received after the channel. The second element is another matrix with the same shape
    (nb_OFDM_symbols, nb_carriers) which is the targeted samples. They are the original samples present before the
    emission in the channel.

    """


    def __init__(self, file_path=None):
        """

        :param file_name:
        """
        # TODO  Make sure to handle the case where the file path doesn't exist.
        if file_path is not None:
            # Load the file
            data_set = np.load(file_path)
            self.samples = data_set(0)
            self.targeted_samples = data_set(1)
        else:
            print("File path undifined")



    def __len__(self):
        """

        :return:
        """


    def __getitem__(self, item):
        """

        :param item:
        :return:
        """

    def save(self, samples, targeted_samples, file_path):
        """
        Save a data set with numpy in a tuple structure.
        :param file_path: A string, the filepath where the file will be located
        :param samples: A N,M-complex array containing the samples at the receiver
        :param targeted_samples: A N,M-complex array containing the targeted sample at the emiter.
        """
        data_set = (samples, targeted_samples)

        with open(file_path, "wb") as handle:
            np.save(handle, data_set)