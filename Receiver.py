import scipy as sp
from scipy import fftpack

import commpy as cp

from Equalizer import switch_init_equalizer
from Utils import *

"""
    Receiver class
"""


class Receiver:
    """ Class of the Receivers

    :cp_len: An integer, length of the cyclic-prefix of the OFDM.
    :nb_carriers: An integer, number of sub_carrier used to transmit data.
    :modulation_order: An integer, the modulation order.
    :nb_off_carriers: An integer, number of off carrier in OFDM scheme
    """

    def __init__(
            self,
            cp_len,
            nb_carriers,
            modulation_order,
            trellis,
            nb_off_carriers=0,
            pilot_frequency=8,
            equalizer_type=None,
    ):
        self.cp_len = cp_len
        self.nb_carriers = nb_carriers
        self.nb_off_carriers = nb_off_carriers
        # Number of carriers that are used knowing the number of off-carrier
        self.nb_on_carriers = self.nb_carriers - self.nb_off_carriers

        # For the pilot management
        self.pilot_frequency = pilot_frequency

        if is_power_of_2(modulation_order):
            self.demodulator = cp.modulation.PSKModem(modulation_order)
        else:
            print("Wrong modulation order : modulation order = ", modulation_order)

        # Generate OFDM pilot symbol to use it for future equalization processes
        pilot_symbols = np.expand_dims(
            self.demodulator.modulate(
                np.zeros(
                    self.nb_on_carriers * int(np.log2(modulation_order)), dtype=int
                )
            ),
            axis=1,
        ).T

        # Test if the trellis is well-defined
        if trellis is not None:
            self.dec_trellis = trellis
        else:
            self.dec_trellis = None
            print("trellis is not defined -> There will have no decoding")

        # Instanciate the Equalizer
        if equalizer_type is not None:
            # Test which type of equalizer
            self.equalizer = switch_init_equalizer(equalizer_type, pilot_symbols)
        else:
            self.equalizer = None

    def demodulate_frame(self, frame, demod_type="hard"):
        """ Demodulate a OFDM received frame
        :frame: A 1D array, received frame to demodulate (following OFDM scheme).
        :demod_type: 'hard'/'soft', type of demodulation wanted.
        """
        # We reshape the frame at the reception
        nb_ofdm_group = len(frame) // (self.cp_len + self.nb_carriers)
        received_frame = np.reshape(
            frame, (nb_ofdm_group, (self.cp_len + self.nb_carriers))
        )
        # We delete the cyclic prefix from each frame
        received_frame = received_frame[:, self.cp_len:]
        # We perform the fft to go from OFDM modulation to standard maping
        time_frame = sp.fftpack.fft(received_frame, axis=1)
        # Then we delete the part of off-carriers
        time_frame = time_frame[:, self.nb_off_carriers:]

        # Every two OFDM symbol, we have to extract the pilots symbols of every two frame
        # and used it for eqaulization
        idx_to_extract = np.arange(0, time_frame.shape[0], self.pilot_frequency + 1)
        received_pilot_symbols = time_frame[idx_to_extract, :]
        #  And delete the pilot symbols
        time_frame = np.delete(time_frame, idx_to_extract, 0)

        # Use of an equalizer ?
        if self.equalizer is not None:
            # Create the new equalized frame which is
            eq_time_frame = np.empty((0, time_frame.shape[1]), dtype=complex)
            # We iterate over the different pilot symbols
            for pilot_idx in range(0, received_pilot_symbols.shape[0]):
                # Set the new estimation
                self.equalizer.estimate(received_pilot_symbols[pilot_idx, :])
                idx = np.arange(
                    self.pilot_frequency * pilot_idx,
                    np.minimum(
                        self.pilot_frequency * (pilot_idx + 1), time_frame.shape[0]
                    ),
                )
                eq_sub_frame = self.equalizer.equalize(time_frame[idx, :])
                # Append the equalized results the new equalized frame
                eq_time_frame = np.append(eq_time_frame, eq_sub_frame, axis=0)
        else:
            # No equalization performed
            eq_time_frame = time_frame

        # Reshape the time frame in a 1D-array
        eq_time_frame = np.ravel(eq_time_frame)

        # Then we perform the demodulation
        return self.demodulator.demodulate(eq_time_frame, demod_type)

    def decode(self, enc_frame):
        """ Decoding an encoded frame according to the trellis of the receiver.
        :enc_frame: A 1D float array, encoded frame to decode.
        """
        if self.dec_trellis is not None :
            # Decode the received frame according to the trellis
            return cp.channelcoding.viterbi_decode(
                enc_frame, self.dec_trellis, decoding_type="hard"  # , tb_length=15
            )
        else:
            return enc_frame