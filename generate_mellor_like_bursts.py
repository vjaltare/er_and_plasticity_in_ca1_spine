import numpy as np
from numpy.random import *

def get_tBurst(beta):
    """returns a Poison spike train with parameter beta = 1/lambda"""
    seed(100)
    t_max = 20 #s
    tBurst = exponential(beta, 1000)
    tBurst = np.cumsum(tBurst)
    tBurst2 = []
    for i in range(tBurst.shape[0]):
        if tBurst[i] < t_max:
            tBurst2.append(tBurst[i])
    tBurst2 = np.insert(tBurst2, 0, 0)
    return np.array(tBurst2)

def get_ISI(tBurst, f_burst_min, f_burst_max, max_inputs):
    """returns the time series for individual spikes.
    Params: tBurst - numpy array of burst times (from get_tBurst function)
            f_burst_min - minimum mean freq of spikes in each burst
            f_burst_max - maximum mean frequency of spikes in each burst
            max_inputs - maximum possible spikes in each burst (~uniform dis [0, max_inputs))
            """
    seed(100)
    #tBurst = get_tBurst(beta_burst)
    t = []
    for i in range(tBurst.shape[0]):
        n_ip = randint(1, max_inputs)
        f_ip = uniform(f_burst_min,f_burst_max)
        #print(f"f={f_ip}")
        #print(f"n={n_ip}")
        t_isi = exponential(1/f_ip, n_ip)
        t_isi = np.cumsum(t_isi)
        #print(t_isi)
        t.extend([tBurst[i] + k for k in t_isi])
    print("end")
    return t