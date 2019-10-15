import numpy as np

class PhyLayer:
    """ PHY layer class
    :param emiter: An Emiter, emiter of the Phy layer
    :param receiver: An Receiver, receiver of the Phy layer
    :param channel: A Channel, which simulate the effect of the channel
    :param feedback_freq: An integer, frequency of the feedback update (if feedback_freq == 0)
    """

    def __init__(self, emitter, receiver, channel, feedback_freq=0):
        self.emitter = emitter
        self.receiver = receiver
        self.channel = channel
        self.feedback_freq = feedback_freq
        self.feedback_counter = 1

    def process_frame(self, frame):
        """
        Apply the sequential operations of the signal's transmission from
        the bit stream of the emitter to the decoded bit stream to the receiver
        :param frame: An integer-array, the bit frame to be emitted.
        :return dec_frame: An a integer-array, the decoded frame at the receiver side.
        """
        # Encode the frame
        enc_frame = self.emitter.encode(frame)
        # Modulate the frame
        mod_frame = self.emitter.modulate_frame(enc_frame)
        # Go through the channel
        channel_frame = self.channel.get_trough(mod_frame)
        # Demodulation of the received frame
        demod_frame = self.receiver.demodulate_frame(channel_frame, demod_type="hard")
        # Shrink for convenience
        demod_frame_shrinked = demod_frame[: len(enc_frame)]
        # Decoding frame
        dec_frame = self.receiver.decode(demod_frame_shrinked)
        # Shrink the last part of the decoded frame before comparing the results
        dec_frame = dec_frame[: len(frame)]
        # Check if we perform a feedback update and
        if (self.feedback_freq != 0):
            # Update the counter
            self.feedback_counter = (self.feedback_counter + 1) % self.feedback_freq
            if (self.feedback_counter == 0):
                # Perform the feedback update
                self.receiver.feedback(dec_frame)
        # Return the decoded frame
        return dec_frame


    def process_frame_ser(self, frame):
        """
        Apply the sequential operations of the signal's transmission from
        the bit stream of the emitter to the decoded bit stream to the receiver
        :param frame: An integer-array, the bit frame to be emitted.
        :return nb_symb_error: An a integer, the number of symbol errors at the receiver side.
        """

        # Encode the frame
        enc_frame = self.emitter.encode(frame)
        # Modulate the frame
        mod_frame = self.emitter.modulate_frame(enc_frame)
        # Go through the channel
        channel_frame = self.channel.get_trough(mod_frame)
        # Demodulation of the received frame
        demod_frame = self.receiver.demodulate_frame(channel_frame, demod_type="hard")
        #print("[PhyLayer.py::process_frame_ser::65] demod_frame ",demod_frame[2045:])
        #print("[PhyLayer.py::process_frame_ser::65] demod_frame shape ",demod_frame.shape)
        # Shrink for convenience
        demod_frame_shrinked = demod_frame[: len(enc_frame)]
        #print("[PhyLayer.py::process_frame_ser::69] demod_frame_shrinked shape", demod_frame_shrinked.shape)
        # Decoding frame
        dec_frame = self.receiver.decode(demod_frame_shrinked)
        #print("[PhyLayer.py::process_frame_ser::72] demod_frame shrinked", demod_frame_shrinked[2045:])
        # Shrink the last part of the decoded frame before comparing the results
        dec_frame_shrinked = dec_frame[: len(frame)]
        #print("[PhyLayer.py::process_frame_ser::74] demod_frame shrinked", demod_frame_shrinked[2045:])
        # Check if we perform a feedback update and
        if (self.feedback_freq != 0):
            # Update the counter
            self.feedback_counter = (self.feedback_counter + 1) % self.feedback_freq
            if (self.feedback_counter == 0):
                # Perform the feedback update
                self.receiver.feedback(dec_frame_shrinked)
        # TODO : optimized the process
        #print("[PhyLayer.py::process_frame_ser::86] No feedback used ")
        mod_frame_fb = self.receiver.raw_mapping(demod_frame_shrinked)
        mod_frame_comp = self.receiver.raw_mapping(enc_frame)

        # Return the decoded frame
        #print("[PhyLayer.py::process_frame_ser::96] frame shape is", frame.shape)
        #print("[PhyLayer.py::process_frame_ser::97] enc_frame shape is ", enc_frame.shape)
        #print("[PhyLayer.py::process_frame_ser::98] mod_frame shape is ", mod_frame.shape)
        #print("[PhyLayer.py::process_frame_ser::99] demod_frame shape is ", demod_frame.shape)
        #print("[PhyLayer.py::process_frame_ser::100] demod_frame_shrinked_shape is ", demod_frame_shrinked.shape)
        #print("[PhyLayer.py::process_frame_ser::101] dec_frame shape is ", dec_frame.shape)
        #print("[PhyLayer.py::process_frame_ser::102] dec_frame_shrinked shape is", dec_frame_shrinked.shape)
        #print("[PhyLayer.py::process_frame_ser::103] mod_frame_fb shape is", mod_frame_fb.shape)
        #print("[PhyLayer.py::process_frame_ser::104] mod_frame_comp shape is", mod_frame_fb.shape)
        #print("[PhyLayer.py::process_frame_ser::105] The number of errors : symbol level is ", np.sum(mod_frame_comp != mod_frame_fb),
        #      " and localized at ", np.argwhere(mod_frame_comp != mod_frame_fb))
        #print("[PhyLayer.py::process_frame_ser::107] The number of errors : coded bit level is", np.sum(enc_frame != demod_frame_shrinked),
        #      " and localized at ", np.argwhere(enc_frame != demod_frame_shrinked))
        #print("[PhyLayer.py::process_frame_ser::109] demod_frame shrinked", demod_frame_shrinked[2045:])
        #print("[PhyLayer.py::process_frame_ser::110] enc_frame", enc_frame[2045:])
        #print("[PhyLayer.py::process_frame_ser::111] The number of errors : at the bit level is", np.sum(frame != dec_frame_shrinked),
        #      " and localized at ", np.argwhere(frame != dec_frame_shrinked))
        #print("\n\n")
        return np.sum(mod_frame_comp != mod_frame_fb), len(mod_frame_fb)