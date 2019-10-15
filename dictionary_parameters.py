"""
Examples of dictionaries used for Monte-Carlo simulation
"""

simulation_param_dict = {
    "m_c_parameters": {
        "min_error_frame": 100, # Number of error frame before getting a BER/FER/SER point
        "targeted_fer": 1e-2, # Targeted FER before stopping simulation
        "step_db": 2, # Step of Eb/N0 for the performances testing
        "min_eb_n0": 0, # First point of performances of Eb/N0
        "max_eb_n0": 50, # Last point of performances of Eb/N0
    },
    "channel_parameters": {
        "channel_type": "Proakis_C", # String to indicate the type of channel that is used
        "non_lin_coeff": 0, # Value of the non-linearity coefficient
        "iq_imbalance": None, # Value of the IQ-imbalance
        "channel_taps": np.array([1, 2, 3, 2, 1]), 
        # Value of the channel taps for the simulation (np.array([1]) can be used if we want to only test AWGN)
    },
    "frame_length": 256, # Frame size
    "modulation": {
        "modulation_order": 4, # Modulation order (PSK modulation)
        "nb_carriers": 64, # Number of OFDM carriers used for modulation
        "cp_length": 8, # Size of the Cyclic Prefix for the OFDM system
        "off_carrier": 0, # Number of off-carriers for the simulation
    },
    "equalizer":"MMSE", # Type of equalizer used (None, ZF, MMSE)
    "channel_coding": { # Parameters for the channel coding part (Handle by CommPy library)
        "mem_size": np.array([2]), # Size of the memory for the channel coding (put it to None if no channel coding)
        "g_matrix": np.array([[0o5, 0o7]]), # Value of the generator polynomial in octal
            # domain (put it to None if no channel coding)
        "rho": 1 / 2, #Coding rate (put 0 if we don't want to use channel coding)
        },
    "pre_equalizer": {
        "model_path": None,
        "feed_back_freq": 0,
    }
}

# TODO used for Monte-Carlo's time step simulation.

simulation_param_dict = {
    "sim_parameters": {
        "nb_frame": 100, # Number of error frame used for each time step
        "eb_n0_db": 2, # Eb/N0 chosen for the performances testing
        "nb_time_step": 1000, # Number of time step used for the simulation
    },
    "channel_parameters": {
        "channel_type": "AWGN", # String to indicate the type of channel that is used
        "non_lin_coeff": 0.5, # Value of the non-linearity coefficient
        "iq_imbalance": np.array([0.45, 0.5, 0.55, 0.6, 0.65]), # Values of the IQ-imbalance
        "chan_param_freq_update": 10, # Update frequency of the iq_imbalance and non-linear coefficient (ie. counter of
            # OFDM symbols send)
        "channel_taps": np.array([1, 2, 3, 2, 1]),
        # Value of the channel taps for the simulation (np.array([1]) can be used if we want to only test AWGN)
    },
    "frame_length": 32, # Frame size (don"t change the value, it was set to fit the article's parameters)
    "modulation": {
        "modulation_order": 4, # Modulation order (PSK moduation)
        "nb_carriers": 64, # Number of OFDM carriers used for modulation
        "cp_length": 8, # Size of the Cyclic Prefix for the OFDM system
        "off_carrier": 0, # Number of off-carriers for the simualtion
    },
    "equalizer":"MMSE", # Type of equalizer used (None, ZF, MMSE)
    "channel_coding": { # Parameters for the channel coding part (Handle by CommPy library)
        "mem_size": np.array([2]), # Size of the memory for the channel coding (put it to None if no channel coding)
        "g_matrix": np.array([[0o5, 0o7]]), # Value of the generator polynomial in octal domain (put it to None if no channel coding)
        "rho": 1 / 2, #Coding rate (put 0 if we don't want to use channel coding)
    },
    "pre_equalizer": {
        "model_path": None,
        "feed_back_freq": 0,
    }
}