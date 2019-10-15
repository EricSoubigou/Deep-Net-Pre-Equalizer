import numpy as np

class SimulationParameters:
    """
    Class of the simulation parameters
    :param _simulation_param_dict:
    """
    # TODO Has to be finished if time.

    def __init__(self, min_error_frame, targeted_fer, step_db, min_eb_n0, max_eb_n0, non_lin_coeff,
                 iq_imbalance, channel_taps, frame_length, modulation_order, nb_carriers, cp_length, off_carrier,
                 equalizer, mem_size, g_matrix, rho, model_path, feed_back_freq):
        self._simulation_param_dict = {
            "m_c_parameters": {
                "min_error_frame": 100,
                "targeted_fer": 5e-3,
                "step_db": 2,
                "min_eb_n0": 0,
                "max_eb_n0": 50,
            },
            "channel_parameters": {
                "non_lin_coeff": 0,
                "iq_imbalance": None,
                "channel_taps": np.array([1]),
            },
            "frame_length": 256,
            "modulation": {
                "modulation_order": 4,
                "nb_carriers": 64,
                "cp_length": 8,
                "off_carrier": 0,
            },
            "equalizer": None, #"ZF", "MMSE"
            "channel_coding": {
                "mem_size": np.array([2]),
                "g_matrix": np.array([[0o5, 0o7]]),
                "rho": 1 / 2,  #  Coding rate
            },
            "pre-equalizer": {
                "model_path": None,
                "feed_back_freq": 10,
            }
        }


    def get_dict

    def set_min_error_frame(self, min_error_frame):
        """
        Set in the dictionary
        :param min_error_frame:
        :return:
        """
