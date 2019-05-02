import numpy as np
import matplotlib.pyplot as plt
import pickle


def is_power_of_2(num):
    """ Return if the number is a power of two.
    :num: An integer, number to test.
    """
    return num != 0 and ((num & (num - 1)) == 0)


def plot_spectrum(signal, time_step):
    """ Plot the power density spectrum of a given signal
    :param signal: A 1D-float-array, with the signal samples
    :param time_step: An integer, time step of the signal sampling
    """
    # Go in the frequency domain
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(signal))) ** 2
    #   f, welch_estimate = sp.signal.welch(signal)
    freq = np.fft.fftshift(np.fft.fftfreq(signal.size, d=time_step))
    plt.plot(freq, spectrum, "r")
    #   plt.plot(f, welch_estimate, 'b')
    plt.yscale("log")
    plt.title("OFDM Spectrogram")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()


def from_real_to_complex(symbols):
    """ Convert a vector of real 2N symbol into a N complex vetor
    with alternate real/imag part of the symbols
    :param symbols: A 2D-real array, containing the value of symbols.
    """
    # Init of the vector
    return symbols[:,0:symbols.shape[1] // 2] + 1j * symbols[:,symbols.shape[1] // 2 :]


def from_complex_to_real(symbols):
    """
    Convert a vector of complex N symbol into a 2N real vetor
    with alternate real/imag part of the symbols
    :param symbols: A 1D-complex array, containing the value of symbols.
    """
    #print("The length of symbols shape is : ",len(symbols.shape))
    if len(symbols.shape) == 1:
        #print("Len 1")
        return np.concatenate((np.real(symbols), np.imag(symbols)))
    else:
        #print("Len ", len(symbols.shape))
        return np.concatenate((np.real(symbols), np.imag(symbols)), axis=1)


def reshape_1D_to_OFDM(cp_length, nb_carriers, frame):
    """
    Reshape the 1D array into a bloc of OFDM symbols
    :param cp_length: A positive integer, the length of the cyclic prefix
    :param nb_carriers:
    :param frame:
    :return:
    """
    nb_ofdm_group = len(frame) // (cp_length + nb_carriers)
    return np.reshape(frame, (nb_ofdm_group, (cp_length + nb_carriers),), )


def load_results(pickle_path):
    """
    Load performances results of a simulation
    :param pickle_path: A string, containing the path to the pickle results file
    :return:
    """
    with open(pickle_path, "rb") as handle:
        return pickle.load(handle)

def plot_ber(eb_n0, ber):
    """
    Plot the performances of a system given its eb_n0 and ber vector.
    :param eb_n0:
    :param ber:
    """
    plt.plot(eb_n0, ber, "r")
    plt.yscale("log")
    plt.title("BER results")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("BER")
    plt.grid(True)
    plt.show()


def plot_performance(dict_list):
    """
    Plot the different performances in dict according a criteria
    :param dict_list: An list, containing all the performances of the
    """
    # Begin the plot
    for idx_plot in range(len(dict_list)):
        print(dict_list[idx_plot])
        if dict_list[idx_plot]["pre-equalizer"]["model_path"] is not None:
            pre_equalizer = "pre_equalizer"
        else:
            pre_equalizer = ""
        plt.plot(dict_list[idx_plot]["results"]["eb_n0_db"],
                 dict_list[idx_plot]["results"]["ber"],
                 label=str(dict_list[idx_plot]["equalizer"] + pre_equalizer))

    plt.yscale("log")
    plt.title("BER results")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("BER")
    plt.grid(True)
    plt.legend()
    plt.show()


def generate_path_name_from_param_dict(sim_param_dict, add_on_path):
    """
    Generate a String being the path for the performances results.
    :param sim_param_dict:
    :param add_on_path:
    :return: A string,
    """
    # Check if there is a pre-equalizer or not
    if sim_param_dict["pre-equalizer"]["model_path"] is not None:
        pre_equalizer = "pre_equalizer_update_" + str(sim_param_dict["pre-equalizer"]["feed_back_freq"])
    else:
        pre_equalizer = ""

    # File name creation
    filename = "./results/OFDM_eq_{}_coding_{}_{}_non_lin_coeff_{}_iq_im_{}_snr_{}_to_{}_step_{}_{}.pickle".format(
        str(sim_param_dict["equalizer"]),
        str(sim_param_dict["channel_coding"]["rho"]),
        sim_param_dict["channel_parameters"]["channel_type"],
        str(sim_param_dict["channel_parameters"]["non_lin_coeff"]),
        str(sim_param_dict["channel_parameters"]["iq_imbalance"]),
        str(sim_param_dict["m_c_parameters"]["min_eb_n0"]),
        str(sim_param_dict["m_c_parameters"]["max_eb_n0"]),
        str(sim_param_dict["m_c_parameters"]["step_db"]),
        pre_equalizer,
        add_on_path,
    )
    return filename