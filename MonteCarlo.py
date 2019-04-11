import pickle
import matplotlib.pyplot as plt

from commpy.channelcoding.convcode import Trellis

from Emiter import *
from Receiver import *
from Channel import *
from PhyLayer import *


def monte_carlo_simulation(sim_param_dict):
    """ Perform the global simulation having the dictionary of parameters.

    :sim_param_dict: A dictionnary containing all parameters necessary for the simualtion
    """

    max_test = (
            sim_param_dict["m_c_parameters"]["min_error_frame"]
            / sim_param_dict["m_c_parameters"]["targeted_fer"]
    )

    #  Simulation parameters
    eb_n0_db = np.array(
        range(
            sim_param_dict["m_c_parameters"]["min_eb_n0"],
            sim_param_dict["m_c_parameters"]["max_eb_n0"],
            sim_param_dict["m_c_parameters"]["step_db"],
        )
    )
    ber = np.zeros(len(eb_n0_db))
    fer = np.zeros(len(eb_n0_db))
    ser = np.zeros(len(eb_n0_db))

    # Compute the snr_db vector
    snr_db = eb_n0_db + 10 * np.log(
        sim_param_dict["channel_coding"]["rho"]
        * np.log2(sim_param_dict["modulation"]["modulation_order"])
    )

    # Creation of the trellis
    trellis = Trellis(
        sim_param_dict["channel_coding"]["mem_size"],
        sim_param_dict["channel_coding"]["g_matrix"],
    )

    # Creation of the emiter
    emiter = Emiter(
        cp_len=sim_param_dict["modulation"]["cp_length"],
        nb_carriers=sim_param_dict["modulation"]["nb_carriers"],
        modulation_order=sim_param_dict["modulation"]["modulation_order"],
        trellis=trellis,
        nb_off_carriers=sim_param_dict["modulation"]["off_carrier"],
    )

    # Creation of the receiver
    receiver = Receiver(
        cp_len=sim_param_dict["modulation"]["cp_length"],
        nb_carriers=sim_param_dict["modulation"]["nb_carriers"],
        modulation_order=sim_param_dict["modulation"]["modulation_order"],
        trellis=trellis,
        nb_off_carriers=sim_param_dict["modulation"]["off_carrier"],
        equalizer_type=sim_param_dict["equalizer"],
    )

    # Creation of the AWGN Channel
    awgn_channel = Channel(
        mean=0,
        var=0,
        non_lin_coeff=sim_param_dict["channel_parameters"]["non_lin_coeff"],
        iq_imbalance=sim_param_dict["channel_parameters"]["iq_imbalance"],
        channel_taps=sim_param_dict["channel_parameters"]["channel_taps"],
    )

    # File name creation
    filename = "./results/OFDM_eq_{}_non_lin_coeff_{}_iq_im_{}_snr_{}_to_{}_step_{}.pickle".format(
        str(sim_param_dict["equalizer"]),
        str(sim_param_dict["channel_parameters"]["non_lin_coeff"]),
        str(sim_param_dict["channel_parameters"]["iq_imbalance"]),
        str(sim_param_dict["m_c_parameters"]["min_eb_n0"]),
        str(sim_param_dict["m_c_parameters"]["max_eb_n0"]),
        str(sim_param_dict["m_c_parameters"]["step_db"]),
    )

    # Creation of the PHY Layer
    phy_layer = PhyLayer(emiter, receiver, awgn_channel)

    nb_tries = 0
    ind_eb_n0 = 0

    # Launch the simulation
    while (ind_eb_n0 < len(eb_n0_db)) and (
            not (ind_eb_n0 > 0) or (ber[ind_eb_n0 - 1] > 0 and nb_tries < max_test)
    ):
        # Init variables
        nb_tries = 0
        nb_frame_error = 0
        global_error_nb = 0
        # Set the snr for the channel
        phy_layer.channel.set_var(
            eb_n0_db[ind_eb_n0], sim_param_dict["modulation"]["modulation_order"]
        )
        # For the moment, we consider that the noise variance is not estimated
        # but is Genie aided.
        if sim_param_dict["equalizer"] == "MMSE":
            receiver.equalizer.set_noise_var(phy_layer.channel.var)
        # Monte-Carlo method

        while (nb_tries < max_test) and (
                nb_frame_error < sim_param_dict["m_c_parameters"]["min_error_frame"]
        ):
            # Generation of the frames
            frame = np.random.randint(0, high=2, size=sim_param_dict["frame_length"])
            # Send the frame to the physical layer
            recieved_frame = phy_layer.process_frame(frame)
            # Counting errors
            errors_num = np.sum(recieved_frame != frame)
            # Look at the number of mistake
            if errors_num > 0:
                # Add the number of frame errors
                nb_frame_error = nb_frame_error + 1
            global_error_nb = global_error_nb + errors_num
            # Increase the number of tries
            nb_tries = nb_tries + 1
        # Update error vectors
        ber[ind_eb_n0] = global_error_nb / (nb_tries * sim_param_dict["frame_length"])
        fer[ind_eb_n0] = nb_frame_error / nb_tries
        #     ser[ind_eb_n0] =
        print(
            "At ",
            np.floor(100 * ind_eb_n0 / len(eb_n0_db)),
            " %",
            ", BER = ",
            ber[ind_eb_n0],
            ", FER = ",
            fer[ind_eb_n0],
            " for  Eb/N0 = ",
            eb_n0_db[ind_eb_n0],
            " dB",
            ", SNR = ",
            snr_db[ind_eb_n0],
            "dB",
            " nb_tries = ",
            nb_tries,
        )
        # Increase the snr index
        ind_eb_n0 += 1
        ber_dict = {
            "sim_param": sim_param_dict,
            "results": {"eb_n0_db": eb_n0_db, "snr_db": snr_db, "ber": ber, "fer": fer},
        }
        # Save results in file
        with open(filename, "wb") as handle:
            pickle.dump(ber_dict, handle)

    # Display results figures
    plt.plot(eb_n0_db, ber, "b")
    plt.yscale("log")
    plt.title("BER results")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("BER")
    plt.grid(True)
    plt.show()

    plt.plot(snr_db, fer, "b")
    plt.yscale("log")
    plt.title("FER results")
    plt.xlabel("SNR (dB)")
    plt.ylabel("FER")
    plt.grid(True)
    plt.show()
