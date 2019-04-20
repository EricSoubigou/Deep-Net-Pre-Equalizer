import torch

from commpy.channelcoding.convcode import Trellis

from Receiver import *
from Emiter import *
from Channel import *

from OFDMSamplesDataset import *


def create_data_set(dataset_param_dict):
    """ To create a data set and save it.
    :param dataset_param_dict :
    """
    # Creation of the trellis
    trellis = Trellis(
        dataset_param_dict["channel_coding"]["mem_size"],
        dataset_param_dict["channel_coding"]["g_matrix"],
    )

    # Creation of the emiter
    emiter = Emiter(
        cp_len=dataset_param_dict["modulation"]["cp_length"],
        nb_carriers=dataset_param_dict["modulation"]["nb_carriers"],
        modulation_order=dataset_param_dict["modulation"]["modulation_order"],
        trellis=trellis,
        nb_off_carriers=dataset_param_dict["modulation"]["off_carrier"],
    )

    # Creation of the receiver
    receiver = Receiver(
        cp_len=dataset_param_dict["modulation"]["cp_length"],
        nb_carriers=dataset_param_dict["modulation"]["nb_carriers"],
        modulation_order=dataset_param_dict["modulation"]["modulation_order"],
        trellis=trellis,
        nb_off_carriers=dataset_param_dict["modulation"]["off_carrier"],
        equalizer_type=dataset_param_dict["equalizer"],
    )

    # Creation of the AWGN Channel
    channel = Channel(
        mean=0,
        var=0,
        non_lin_coeff=dataset_param_dict["channel_parameters"]["non_lin_coeff"],
        iq_imbalance=dataset_param_dict["channel_parameters"]["iq_imbalance"],
        channel_taps=dataset_param_dict["channel_parameters"]["channel_taps"],
    )

    # Set of the variable
    channel.set_var(
        dataset_param_dict["eb_n0_db"],
        dataset_param_dict["modulation"]["modulation_order"],
    )

    # Generation of the frames
    frame = np.random.randint(0, high=2, size=dataset_param_dict["frame_length"])
    # Encode the frame
    enc_frame = emiter.encode(frame)
    # Modulate the frame
    mod_frame = emiter.modulate_frame(enc_frame)
    # Go through the channel
    channel_frame = channel.get_trough(mod_frame)

    samples = reshape_1D_to_OFDM(dataset_param_dict["modulation"]["cp_length"],
                                 dataset_param_dict["modulation"]["nb_carriers"],
                                 mod_frame)
    targets = reshape_1D_to_OFDM(dataset_param_dict["modulation"]["cp_length"],
                                 dataset_param_dict["modulation"]["nb_carriers"],
                                 channel_frame)

    # Save results in file
    filename = "./data_set/OFDM_non_lin_coeff_{}_iq_im_{}_eb_n0_{}_proakis_C.pt".format(
        str(dataset_param_dict["channel_parameters"]["non_lin_coeff"]),
        str(dataset_param_dict["channel_parameters"]["iq_imbalance"]),
        str(dataset_param_dict["eb_n0_db"]),
    )

    OFDMSamplesDataset.save(samples, targets, filename)



