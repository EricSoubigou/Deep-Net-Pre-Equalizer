

class PhyLayer:
    """ PHY layer class
    :emiter: An Emiter, emiter of the Phy layer
    :receiver: An Receiver, receiver of the Phy layer
    :channel: A Channel,
    """

    def __init__(self, emiter, receiver, channel):
        self.emiter = emiter
        self.receiver = receiver
        self.channel = channel

    def process_frame(self, frame):
        # Encode the frame
        enc_frame = self.emiter.encode(frame)
        # Modulate the frame
        mod_frame = self.emiter.modulate_frame(enc_frame)
        # Go through the channel
        channel_frame = self.channel.get_trough(mod_frame)
        # Demodulation of the received frame
        demod_frame = self.receiver.demodulate_frame(channel_frame, demod_type="hard")
        # Shrink for convinience
        demod_frame = demod_frame[: len(enc_frame)]
        # Decoding frame
        dec_frame = self.receiver.decode(demod_frame)
        # Shrink the last part of the decoded frame before comparing the results
        return dec_frame[: len(frame)]