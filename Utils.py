import numpy as np
import matplotlib.pyplot as plt


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
    # cmplx = np.array((1, np.round(len(symbols) / 2)), dtype=complex)
    return symbols[0:2:] + 1j * symbols[1:2:]


def from_complex_to_real(symbols):
    """
    Convert a vector of complex N symbol into a 2N real vetor
    with alternate real/imag part of the symbols
    :param symbols: A 1D-complex array, containing the value of symbols.
    """
    print("cat shape", np.concatenate((np.real(symbols), np.imag(symbols)), axis=1).shape)
    print('separated vector', np.real(symbols).shape, np.imag(symbols).shape)
    x = np.ravel(np.concatenate((np.real(symbols), np.imag(symbols)), axis=1))
    print("x shape", x.shape)
    return x


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
