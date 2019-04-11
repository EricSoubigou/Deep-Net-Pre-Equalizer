import torch

from commpy.channelcoding.convcode import Trellis

from Receiver import *
from Emiter import *
from Channel import *

def create_data_set(dataset_param_dict):
    """ To create a data set and save it.
    ::
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
    # Demodulation of the received frame
    demod_frame = receiver.demodulate_frame(channel_frame, demod_type="hard")

    # Since we get through the channel, we formate the data for a tensor
    # We reshape the frame at the reception
    nb_ofdm_group = len(mod_frame) // (
            dataset_param_dict["modulation"]["cp_length"]
            + dataset_param_dict["modulation"]["nb_carriers"]
    )
    target_samples = np.reshape(
        mod_frame,
        (
            nb_ofdm_group,
            (
                    dataset_param_dict["modulation"]["cp_length"]
                    + dataset_param_dict["modulation"]["nb_carriers"]
            ),
        ),
    )



    # Convert complex values into real and cast into a tensor
    target_samples = torch.from_numpy(from_complex_to_real(target_samples))
    samples = torch.from_numpy(from_complex_to_real(demod_frame))

    # Save results in file
    filename = "./data_set/OFDM_non_lin_coeff_{}_iq_im_{}_eb_n0_{}_proakis_C.pt".format(
        str(dataset_param_dict["channel_parameters"]["non_lin_coeff"]),
        str(dataset_param_dict["channel_parameters"]["iq_imbalance"]),
        str(dataset_param_dict["eb_n0_db"]),
    )

    torch.save(target_samples, filename)

    print("Data set created at " + filename)

###

data_set_generation_param_dict = {
    "eb_n0_db": 10,
    "channel_parameters": {
        "non_lin_coeff": 0,
        "iq_imbalance": None,
        "channel_taps": np.array([1, 2, 3, 2, 1]),
    },
    "frame_length": 1000,
    "modulation": {
        "modulation_order": 4,
        "nb_carriers": 64,
        "cp_length": 8,
        "off_carrier": 0,
    },
    "equalizer": "MMSE",
    "channel_coding": {
        "mem_size": np.array([2]),
        "g_matrix": np.array([[0o5, 0o7]]),
        "rho": 1 / 2,  #  Coding rate
    },
}

if __name__ == '__main__':
    create_data_set(data_set_generation_param_dict)