import numpy as np
import scipy as sp

from scipy import signal


class Channel:

    """ Class of the AWGN channel.
    :mean: A float, mean of the gaussian noise.
    :var: A float, variance of the gaussian noise.
    :non_lin_coeff: A float, value of the non-linearity coefficient of the channel.
    :iq_imbalance: A float, value of the iq_imbalance.
    :channel_taps: A 1-D float-array, containing the value of channel's taps.
    :up_factor: A positive integer, value of the upsampling factor
    """

    def __init__(
        self,
        mean,
        var,
        non_lin_coeff=0,
        iq_imbalance=None,
        channel_taps=np.ones(1),
        up_factor=1,
    ):
        self.mean = mean
        self.var = var
        self.gamma = non_lin_coeff
        self.beta = iq_imbalance
        self.channel_taps = np.divide(channel_taps, np.linalg.norm(channel_taps))
        self.up_factor = up_factor

    def get_trough(self, mod_frame):
        """ Return the value of the frame after get through the channel
        :mod_frame: An array, input of of the channel. Is the modulated frame from an
        emiter.
        """
        # Add the IQ imbalance at the emiter side
        mod_frame = self.add_iq_imbalance(mod_frame)
        # Add the non linearity effect at the emiter side
        mod_frame -= self.gamma * (np.abs(mod_frame)) ** 2 * mod_frame
        # Go through the multipath channel if the response length is greater than one tap
        if len(self.channel_taps) > 1:
            # Perform the channel filtering
#             print("mod_frame before up-sampling", mod_frame.shape)
#             print("Mod frame output: ", mod_frame[6:20])
            #mod_frame = sp.signal.resample(mod_frame, len(mod_frame)*self.up_factor)
#             print("mod_frame after up-sampling", mod_frame.shape)
#             print("Mod frame output: ", mod_frame[6:20])
            # Filter
#             print("Channel taps", self.channel_taps)
            mod_frame = sp.signal.lfilter(b=self.channel_taps, a=[1], x=mod_frame)
#             print("mod_frame after filtering", mod_frame.shape)
#             print("Mod frame output: ", mod_frame[6:20])
            # Down sample the signal
            #mod_frame = sp.signal.decimate(mod_frame, q=self.up_factor)
#             print("mod_frame after decimation", mod_frame.shape)
#             print("Mod frame output: ", mod_frame[6:20])
#             mod_frame = sp.signal.upfirdn(
#                 self.channel_taps, mod_frame, up=self.up_factor, down=self.up_factor
#             )
            # Then shrink the reponse
            #mod_frame = mod_frame[len(self.channel_taps)-1:]
        # Add Gaussian noise
        output = mod_frame + np.random.normal(self.mean, self.var, mod_frame.shape)
        # Add the second IQ imbalance at the receiver side
        output = self.add_iq_imbalance(output)
        # Add the non linearity effect at the receiver side
        output -= self.gamma * np.abs(output) ** 2 * output
        return output

    def add_iq_imbalance(self, x):
        """ Add IQ imbalance to a given array
        :x: An array, input which will be imbalanced according to attributes of the
        channel class
        """
        if self.beta is not None:
            return self.beta * np.real(x) + 1j * np.imag(x)
        else:
            return x

    def set_var(self, snr_db, modulation_order):
        """ Set the variance of the gaussian noise of the channel
        :snr_db: A float, Value of the Signal to Noise ratio in dB
        :modulation_order: An integer, is the moudlation order of the constellation used
        """
        snr = 10 ** (snr_db / 10)
        # To define a bit better than it is right now
        var_signal = 1
        shaped_signal = 1
        self.var = (
            np.linalg.norm(shaped_signal)
            * var_signal
            * np.linalg.norm(self.channel_taps)
        ) / (2 * np.log2(modulation_order) * snr)