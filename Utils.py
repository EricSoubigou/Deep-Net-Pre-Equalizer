import matplotlib as plt
import numpy as np


""" Return if the number is a power of two. 
:num: An integer, number to test.
"""
def is_power_of_2(num):
    return num != 0 and ((num & (num - 1)) == 0)

""" Plot the power density spectrum of a given signal
:signal: A 1D-float-array, with the signal samples
:time_step: An integer, time step of the signal sampling 
"""
def plot_spectrum(signal, time_step):
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
    :symbols: A 2D-real array, containing the value of symbols.
    """
    # Init of the vector
    cmplx = np.array((1, np.round(len(symbols) / 2)), dtype=complex)
    cmplx = symbols[0:2:] + j * symbols[1:2:]
    return cmplx

def from_complex_to_real(symbols):
    """ Convert a vector of complex N symbol into a 2N real vetor
    with alternate real/imag part of the symbols
    :symbols: A 1D-complex array, containing the value of symbols.
    """
    return np.ravel(np.concatenate((np.real(symbols), np.imag(symbols)), axis=1))