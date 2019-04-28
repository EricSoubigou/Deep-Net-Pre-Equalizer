"""
  Emitter class
"""

import commpy as cp
import copy as cpy

from scipy import fftpack

from Utils import *


class Emiter:
    """ Class of the emiter.

  :cp_len: An integer, length of the cyclic-prefix of the OFDM.
  :nb_carriers: An integer, number of sub_carrier used to transmit data.
  :modulation_order: An integer, the modulation order.
  :nb_off_carriers: An integer, number of off carrier in OFDM scheme
  :trellis: A compy.channelcoding.Trellis, to encode the data
  :pilot_frequency: A integer, number of OFDM symbols/frame before transmitting a pilot

  Ex : | CP | CP | OFF | OFF | ON | ON | ... | ON |
  """

    def __init__(
            self,
            cp_len,
            nb_carriers,
            modulation_order,
            trellis,
            nb_off_carriers=0,
            pilot_frequency=8,
    ):
        self.cp_len = cp_len
        self.nb_carriers = nb_carriers
        self.nb_off_carriers = nb_off_carriers
        # Number of carriers that are used knowing the number of off-carrier
        self.nb_on_carriers = self.nb_carriers - self.nb_off_carriers

        # For the pilot management
        self.pilot_frequency = pilot_frequency

        if is_power_of_2(modulation_order):
            self.modulator = cp.modulation.PSKModem(modulation_order)
        else:
            print("Wrong modulation order : modulation order = ", modulation_order)

        # Generate OFDM pilot symbol
        self._pilot_symbols = np.expand_dims(
            self.modulator.modulate(
                np.zeros(
                    self.nb_on_carriers * int(np.log2(modulation_order)), dtype=int
                )
            ),
            axis=1,
        ).T

        # Test if the trellis is well-defined
        if trellis is not None:
            self.enc_trellis = trellis
        else:
            self.enc_trellis = None
            print("Trellis is not defined -> There will have no encoding")

    def get_trellis(self):
        """ Return the copy of trellis of the emiter
        """
        return cpy.deepcopy(self.enc_trellis)

    def modulate_frame(self, frame):
        """ Modulate and Map the frame. In other words, will perform the
        Modulation and the Mapping and then perform the OFDM transmormation before
        sending the data.
        :frame: The frame that has to be modulated.
        """
        # Mapping of the data
        mod_frame = self.modulator.modulate(frame)

        # Test if the division is equal to an integer or not
        if len(mod_frame) % self.nb_on_carriers != 0:
            nb_ofdm_group = (len(mod_frame) // self.nb_on_carriers) + 1
            # Add padding to the frame in order to get a interger number of PHY
            # frames used.

            padding = np.zeros(
                (self.nb_on_carriers - len(mod_frame) % self.nb_on_carriers)
                * self.modulator.num_bits_symbol,
                dtype=int,
            )
            # Add padding to the modulated frame
            mod_frame = np.concatenate(
                (mod_frame, self.modulator.modulate(padding)), axis=0
            )
        else:
            # No padding to perform
            nb_ofdm_group = len(mod_frame) // self.nb_on_carriers

        # Then reshape the frame to perform the modulation
        carriers = np.reshape(mod_frame, (nb_ofdm_group, self.nb_on_carriers))

        # Insert the pilot symbols at the given frequency along rows
        carriers = np.insert(
            carriers,
            np.arange(0, carriers.shape[0], self.pilot_frequency),
            self._pilot_symbols,
            axis=0,
        )

        # Test if there are some off carriers
        if self.nb_off_carriers > 0:
            # Add the off_carriers
            carriers = np.concatenate(
                (np.zeros((nb_ofdm_group, self.nb_off_carriers)), carriers), axis=1
            )

        # Then use the matrix to transform them into OFDM symbols
        ofdm_signal = fftpack.ifft(carriers, axis=1)
        #  Add the cyclic prefix
        ofdm_signal_cp = np.concatenate(
            (ofdm_signal[:, self.nb_carriers - self.cp_len:], ofdm_signal), axis=1
        )
        # Return the global modulated frame
        return np.ravel(ofdm_signal_cp)

    def encode(self, frame):
        """ Encode the bit frame according to the defined trellis of the emiter
        :frame: The frame that has to be encoded.
        """
        # Channel coding of the frame if there is a trellis
        if self.enc_trellis is not None:
            return cp.channelcoding.conv_encode(frame, self.enc_trellis)
        else:
            # In the case where there is no coding
            return frame
