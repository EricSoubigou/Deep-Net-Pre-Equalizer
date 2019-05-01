"""
    Equalizer class
"""
import numpy as np


class Equalizer:
    """ Class of Equalizers
    :pilot_symbols: An 1D-float array, beeing the pilots symbol used to estimate
    """

    def __init__(self, pilot_symbols):
        self.pilot_symbols = pilot_symbols
        self.estimation = None
        self._name = "None"

    def equalize(self, symbols_to_equalize):
        """ Equalizes the received frame
        :param symbols_to_equalize: A 2D-float-array, with the symbol to equalize
        """
        return np.divide(symbols_to_equalize, self.estimation)

    def get_name(self):
        return self._name


class ZeroForcing(Equalizer):
    """ Zero Forcing equalizers class
    """

    def __init__(self, pilot_symbols):
        super().__init__(pilot_symbols)
        self._name = "Zero-Forcing"

    def estimate(self, received_pilot_symbols):
        """
        :parma received_pilot_symbols: A 1D-complex-array, containing the signal samples of the
            received pilot symbols.
        """
        self.estimation = np.divide(
            np.multiply(received_pilot_symbols, np.conjugate(self.pilot_symbols)),
            np.power(np.abs(self.pilot_symbols), 2),
        )

    def equalize(self, symbols_to_equalize):
        """ Equalizes the received frame
        :param symbols_to_equalize: A 2D-float-array, with the symbol to equalize
        """
        return np.divide(
            np.multiply(symbols_to_equalize, np.conjugate(self.estimation)),
            np.power(np.abs(self.estimation), 2),
        )


class MMSE(Equalizer):
    """ MMSE equalizers class
    :param noise_var_est: A float, value of the gaussian noise variance.
    """

    def __init__(self, pilot_symbols):
        super().__init__(pilot_symbols)
        self._name = "MMSE"
        self._noise_var_est = 0

    def estimate(self, received_pilot_symbols):
        """ Estimate the channel taps following the MMSE principles
        :param received_pilot_symbols: A 1D-complex-array, containing the signal samples of the
            received pilot symbols.
        """
        self.estimation = np.divide(
            np.multiply(received_pilot_symbols, np.conjugate(self.pilot_symbols)),
            np.linalg.norm(self.pilot_symbols) ** 2 + self._noise_var_est,
        )

    def set_noise_var(self, noise_var_est):
        """ Set the value of the gaussian noise variance
        :param noise_var_est: A float, value of the noise variance estimated.
        """
        self._noise_var_est = noise_var_est

    def equalize(self, symbols_to_equalize):
        """ Equalizes the received frame
        :param symbols_to_equalize: A 2D-float-array, with the symbol to equalize
        """

        return np.divide(
            np.multiply(
                symbols_to_equalize,
                np.conjugate(self.estimation)
            ),
            np.linalg.norm(self.estimation) ** 2
            + self._noise_var_est,
        )


def switch_init_equalizer(equalizer_name, pilot_symbols):
    """ Instantiate the wanted equalizer given the name of the equalizer
    """
    equalizers = {
        "ZF": ZeroForcing(pilot_symbols=pilot_symbols),
        "MMSE": MMSE(pilot_symbols=pilot_symbols),
    }
    return equalizers.get(equalizer_name, None)
