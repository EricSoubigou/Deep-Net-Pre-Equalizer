

class PhyLayer:
    """ PHY layer class
    :param emiter: An Emiter, emiter of the Phy layer
    :param receiver: An Receiver, receiver of the Phy layer
    :param channel: A Channel, which simulate the effect of the channel
    :param feedback_freq: An integer, frequency of the feedback update (if feedback_freq == 0)
    """

    def __init__(self, emiter, receiver, channel, feedback_freq=0):
        self.emiter = emiter
        self.receiver = receiver
        self.channel = channel
        self.feedback_freq = feedback_freq
        self.feedback_counter = 0

    def process_frame(self, frame):
        """
        Apply the sequential operations of the signal's transmission from
        the bit stream of the emitter to the decoded bit stream to the receiver
        :param frame:
        :return:
        """
        # Encode the frame
        enc_frame = self.emiter.encode(frame)
        # Modulate the frame
        mod_frame = self.emiter.modulate_frame(enc_frame)
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
        if (self.feedback_freq != 0) and (self.feedback_counter == 0):
            # Update the counter
            self.feedback_counter = (self.feedback_counter + 1) % self.feedback_freq
            # Perform the feedback update
            dec_frame = self.receiver.feedback(dec_frame)
        # Return the decoded frame
        return dec_frame