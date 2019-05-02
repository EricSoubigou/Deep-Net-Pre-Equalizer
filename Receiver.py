import scipy as sp
import torch
from scipy import fftpack

from copy import deepcopy

import commpy as cp

from Equalizer import switch_init_equalizer
from Utils import *

from PreEqualizer import PreEqualizer

class Receiver:
    """
    Class of the Receivers
    :param cp_len: An integer, length of the cyclic-prefix of the OFDM.
    :param nb_carriers: An integer, number of sub_carrier used to transmit data.
    :param modulation_order: An integer, the modulation order.
    :param nb_off_carriers: An integer, number of off-carrier in OFDM scheme
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
            pre_equalizer_path=None,
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
        self._pilot_symbols = np.expand_dims(
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

        # Instantiate the Equalizer
        if equalizer_type is not None:
            # Test which type of equalizer
            self.equalizer = switch_init_equalizer(equalizer_type, self._pilot_symbols)
        else:
            self.equalizer = None

        # Instantiate the Pre-Equalizer
        if pre_equalizer_path is not None:
            self.pre_equalizer = PreEqualizer(self.nb_carriers + self.cp_len)
            self.pre_eq_out = None # To specify
            try:
                self.pre_equalizer.load_state_dict(torch.load(pre_equalizer_path))
            except:
                print("Can't load {} => Generate a new model (random)".format(pre_equalizer_path))
        else:
            self.pre_equalizer = None

    def demodulate_frame(self, frame, demod_type="hard"):
        """
        Demodulate a OFDM received frame
        :param frame: A 1D array, received frame to demodulate (following OFDM scheme).
        :param demod_type: 'hard'/'soft', type of demodulation wanted.
        """
        # We reshape the frame at the reception
        nb_ofdm_group = len(frame) // (self.cp_len + self.nb_carriers)
        received_frame = np.reshape(
            frame, (nb_ofdm_group, (self.cp_len + self.nb_carriers))
        )
        #print("received frame type is ", received_frame.shape)
        #print("converted frame type is ", from_complex_to_real(received_frame[1, :]).shape)

        # We pre-equalize if possible
        if self.pre_equalizer is not None:
            # Iterate over the data set
            self.pre_eq_out = torch.Tensor(nb_ofdm_group, 2 * (self.nb_carriers + self.cp_len)).float()
            # Cast the data into a 2 times real array and convert it into a Floar
            # tensor to use it with the pre_equalizer
            torch_frame = torch.from_numpy(from_complex_to_real(received_frame)).float()
            # Perform the pre-equalization.
            self.pre_eq_out = self.pre_equalizer(torch_frame)
            # Clone the output (to feedforward without the gradient problem due to Pytorch architecture)
            out = self.pre_eq_out.clone()
            # Detach the gradient from the clones
            out_numpy = torch.Tensor.numpy(out.detach())
            # Get back in the complex domain
            received_frame = from_real_to_complex(out_numpy)

        # We delete the cyclic prefix from each frame
        received_frame = received_frame[:, self.cp_len:]
        # We perform the fft to go from OFDM modulation to standard mapping
        time_frame = sp.fftpack.fft(received_frame, axis=1)
        # Then we delete the part of off-carriers
        time_frame = time_frame[:, self.nb_off_carriers:]

        # Every  self.pilot_frequency OFDM symbol, we have to extract the pilots symbols of every two frame
        # and used it for equalization
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
        """
        Decoding an encoded frame according to the trellis of the receiver.
        :param enc_frame: A 1D float array, encoded frame to decode.
        """
        if self.dec_trellis is not None:
            # Decode the received frame according to the trellis
            return cp.channelcoding.viterbi_decode(
                enc_frame, self.dec_trellis, decoding_type="hard"  # , tb_length=15
            )
        else:
            return enc_frame

    def encode(self, frame):
        """
        Encode the bit frame according to the defined trellis of the receiver. This function has been created to
        allow the receiver to perform the feeback update.
        :param frame: The frame that has to be encoded.
        """
        # Channel coding of the frame if there is a trellis
        if self.dec_trellis is not None:
            return cp.channelcoding.conv_encode(frame, self.dec_trellis)
        else:
            # In the case where there is no coding
            return frame

    def modulate_frame(self, frame):
        """
        Modulate and Map the frame. In other words, will perform the
        Modulation and the Mapping and then perform the OFDM transformation before
        sending the data. This function was developed to perform the feedback loop and get the inverse transform
        :param frame: The frame that has to be modulated.
        """
        # Mapping of the data
        mod_frame = self.demodulator.modulate(frame)

        # Test if the division is equal to an integer or not
        if len(mod_frame) % self.nb_on_carriers != 0:
            nb_ofdm_group = (len(mod_frame) // self.nb_on_carriers) + 1
            # Add padding to the frame in order to get a integer number of PHY
            # frames used.

            padding = np.zeros(
                (self.nb_on_carriers - len(mod_frame) % self.nb_on_carriers)
                * self.demodulator.num_bits_symbol,
                dtype=int,
            )
            # Add padding to the modulated frame
            mod_frame = np.concatenate(
                (mod_frame, self.demodulator.modulate(padding)), axis=0
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

        # Perform the de-equalization of the frame.
        if self.equalizer is not None:
            # Nota : The estimation is the same => would be inaccurate if the channel was time-variant.
            carriers = self.equalizer.de_equalize(carriers)
            # Append the equalized results the new equalized frame
        else:
            # No equalization performed
            carriers = carriers


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
        # Return the ofdm signal in block
        return ofdm_signal_cp


    def feedback(self, est_frame_dec):
        """
        Perform the feedback operation to update the network given the Viterbi decoding information
        :param est_frame_dec: An array, decoded frame.
        """
        # Check that the pre_equalizer is instantiate for the receiver
        assert self.pre_equalizer is not None, "PreEqualizer is not instantiate for pre_equalizer for the Receiver"
        # Go from the decoded space to the encoded space
        est_frame_dec_enc = self.encode(est_frame_dec)
        # Redo the whole chain in the opposite way:
        est_frame_dec_enc_mod = self.modulate_frame(est_frame_dec_enc)
        # Perform the feedback update
        self.pre_equalizer.feedback_update(self.pre_eq_out, est_frame_dec_enc_mod)


