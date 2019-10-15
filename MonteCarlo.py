from commpy.channelcoding.convcode import Trellis

from Emitter import *
from Receiver import *
from Channel import *
from PhyLayer import *


def monte_carlo_simulation(sim_param_dict, add_on_path=""):
    """ Perform the global simulation having the dictionary of parameters.

    :param sim_param_dict: A dictionary containing all parameters necessary
        for the simulation
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
    if sim_param_dict["channel_coding"]["rho"] != 1:
        # print("rho value is ", sim_param_dict["channel_coding"]["rho"])
        trellis = Trellis(
            sim_param_dict["channel_coding"]["mem_size"],
            sim_param_dict["channel_coding"]["g_matrix"],
        )
    else:
        # No coding
        trellis = None

    # Creation of the emiter
    emiter = Emitter(
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
        pre_equalizer_path=sim_param_dict["pre_equalizer"]["model_path"],
    )

    # Creation of the AWGN Channel
    awgn_channel = Channel(
        mean=0,
        var=0,
        non_lin_coeff=sim_param_dict["channel_parameters"]["non_lin_coeff"],
        iq_imbalance=sim_param_dict["channel_parameters"]["iq_imbalance"],
        channel_taps=sim_param_dict["channel_parameters"]["channel_taps"],
    )

    # Creation of the filepath
    filename = generate_path_name_from_param_dict(sim_param_dict, add_on_path)
    print("Results will be printed in : ", filename)

    # Creation of the PHY Layer
    phy_layer = PhyLayer(emiter, receiver, awgn_channel, sim_param_dict["pre_equalizer"]["feed_back_freq"])

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

    # TODO : Use the function of utils when they will be well-implemented
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


def feedback_update_simulation(sim_param_dict, add_on_path=""):
    """
    Perform simulation having the dictionary of parameters.
    This one is used on the case of time step study and the performances of feedback loop

    :param sim_param_dict: A dictionary containing all parameters necessary
        for the simulation
    """

    #  Simulation parameters
    eb_n0_db = sim_param_dict["sim_parameters"]["eb_n0_db"]

    nb_time_step = sim_param_dict["sim_parameters"]["nb_time_step"]
    chan_freq_update = sim_param_dict["channel_parameters"]["chan_param_freq_update"]

    time_array = np.linspace(0, nb_time_step, nb_time_step)
    ser = np.zeros(nb_time_step)
    gamma_values = np.zeros(nb_time_step)
    beta_values = np.zeros(nb_time_step)

    # Compute the snr_db value.
    snr_db = eb_n0_db + 10 * np.log(
        sim_param_dict["channel_coding"]["rho"]
        * np.log2(sim_param_dict["modulation"]["modulation_order"])
    )

    # Creation of the trellis
    if sim_param_dict["channel_coding"]["rho"] != 1:
        # print("rho value is ", sim_param_dict["channel_coding"]["rho"])
        trellis = Trellis(
            sim_param_dict["channel_coding"]["mem_size"],
            sim_param_dict["channel_coding"]["g_matrix"],
        )
    else:
        # No coding
        trellis = None

    # Creation of the emitter
    emiter = Emitter(
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
        pre_equalizer_path=sim_param_dict["pre_equalizer"]["model_path"],
    )

    print((sim_param_dict["channel_parameters"]["non_lin_coeff"]))
    # Creation of the AWGN Channel
    awgn_channel = Channel(
        mean=0,
        var=0,
        non_lin_coeff=sim_param_dict["channel_parameters"]["non_lin_coeff"],
        iq_imbalance=sim_param_dict["channel_parameters"]["iq_imbalance"],
        # Not sure that this will work : have to test in the case of a non array type
        channel_taps=sim_param_dict["channel_parameters"]["channel_taps"],
    )

    # Creation of the filepath
    filename = generate_path_name_from_param_dict_ser(sim_param_dict, add_on_path)
    print("Results will be printed in : ", filename)

    # Creation of the PHY Layer
    phy_layer = PhyLayer(emiter, receiver, awgn_channel, sim_param_dict["pre_equalizer"]["feed_back_freq"])

    # Set the snr for the channel for once.
    phy_layer.channel.set_var(
        eb_n0_db, sim_param_dict["modulation"]["modulation_order"]
    )
    # For the moment, we consider that the noise variance is not estimated
    # but is Genie aided.
    if sim_param_dict["equalizer"] == "MMSE":
        receiver.equalizer.set_noise_var(phy_layer.channel.var)

    # Parameters of the iteration process.
    ind_time_step = 0

    # Launch the simulation while we do not fill the vector of time steps.
    while (ind_time_step < nb_time_step):
        # Record the value of gamma and iq
        gamma_values[ind_time_step] = phy_layer.channel.gamma
        beta_values[ind_time_step] = phy_layer.channel.beta
        # Generation of the frames
        frame = np.random.randint(0, high=2, size=sim_param_dict["frame_length"])
        # Send the frame to the physical layer and get the number of errors at the symbol level
        errors_num, nb_symb = phy_layer.process_frame_ser(frame)
        # If we have reached the frequency threshhold we update the value of
        if np.remainder(ind_time_step, chan_freq_update) == 0:
            # Select a new random value of non-linearity
            phy_layer.channel.random_update_non_lin(sim_param_dict["channel_parameters"]["non_lin_coeff_set"],
                                                    sim_param_dict["channel_parameters"]["iq_imbalance_set"])
            print("New parameter gamma is ",
                phy_layer.channel.gamma,
                "beta is ",
                phy_layer.channel.beta,
                " at t = ",
                ind_time_step)
        # Update error vectors.
        ser[ind_time_step] = errors_num / nb_symb
        if ((ind_time_step % 100) == 0):
            #print("[MonteCarlo.py::feedback_update_simulation::276]The number of symbols is :", nb_symb)
            print(
                "At ",
                np.floor(100 * ind_time_step / nb_time_step),
                " %",
                ", SER = ",
                ser[ind_time_step],
                " for  Eb/N0 = ",
                eb_n0_db,
                " dB",
                ", SNR = ",
                snr_db,
                "dB",
                " at time step",
                ind_time_step,
            )
            ber_dict = {
                "sim_param": sim_param_dict,
                "results": {
                    "eb_n0_db": eb_n0_db,
                    "snr_db": snr_db,
                    "time_array": time_array,
                    "ser": ser,
                    "gamma_values":gamma_values,
                    "beta_values": beta_values,
                },
            }
        # Save results in file
        with open(filename, "wb") as handle:
            pickle.dump(ber_dict, handle)
        # Increase the time step
        ind_time_step += 1

    # Display results figures TODO : Implements a function that performs that in Utils.
    plt.plot(time_array, ser, "b")
    plt.yscale("log")
    plt.title("BER results")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("SER")
    plt.grid(True)
    plt.show()


