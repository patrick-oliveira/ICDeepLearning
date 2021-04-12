from Project import Project

import numpy as np
from sklearn.cross_decomposition import CCA
import os
from pathlib import Path
import ray

def normalize(X: np.array, varAxis: int = 1) -> np.array:
    '''
    varAxis 
    '''
    return (X - np.mean(X, axis = 0))/np.std(X, axis = 0) if varAxis == 1 else (X - np.mean(X, axis = 1))/np.std(X, axis = 1)

def stimulus(f : [float], phi : [float], length : int, k = 60) -> np.array:
    return np.array([0.5*( 1 + np.sin(2*np.pi*f*(i/k) + phi) ) for i in range(length)])

def referenceSignal(f: float, Nh: int, Fs: int, N: int) -> np.array:
    '''
    f: reference frequency
    Nh: number of harmonics
    Fs: sampling rate
    N: number of time samples
    '''
    t = np.asarray(range(1, N + 1))/Fs
    return np.vstack(np.asarray([[np.sin(2*np.pi*Ni*f*t), 
                                  np.cos(2*np.pi*Ni*f*t)] for Ni in range(1, Nh + 1)]))


@ray.remote
def part1():
    for C in range(1, 10 + 1):
        for file in os.listdir(base_path / str(C)):
            X = np.loadtxt((base_path / str(C) / file).__str__())
            ca.fit(X, Y)
            X_c, Y_c = ca.transform(X, Y)
            (outp_path / str(C)).mkdir(parents = True, exist_ok = True)
            np.savetxt((outp_path / str(C) / file).__str__(), X_c)

@ray.remote
def part2():
    for C in range(11, 20 + 1):
        for file in os.listdir(base_path / str(C)):
            X = np.loadtxt((base_path / str(C) / file).__str__())
            ca.fit(X, Y)
            X_c, Y_c = ca.transform(X, Y)
            (outp_path / str(C)).mkdir(parents = True, exist_ok = True)
            np.savetxt((outp_path / str(C) / file).__str__(), X_c)
            
# @ray.remote
# def part3():
#     for C in range(21, 30 + 1):
#         for file in os.listdir(base_path / str(C)):
#             X = np.loadtxt((base_path / str(C) / file).__str__())
#             ca.fit(X, Y)
#             X_c, Y_c = ca.transform(X, Y)
#             (outp_path / str(C)).mkdir(parents = True, exist_ok = True)
#             np.savetxt((outp_path / str(C) / file).__str__(), X_c)
            
# @ray.remote
# def part4():
#     for C in range(31, 40 + 1):
#         for file in os.listdir(base_path / str(C)):
#             X = np.loadtxt((base_path / str(C) / file).__str__())
#             ca.fit(X, Y)
#             X_c, Y_c = ca.transform(X, Y)
#             (outp_path / str(C)).mkdir(parents = True, exist_ok = True)
#             np.savetxt((outp_path / str(C) / file).__str__(), X_c)
            
if __name__ == '__main__':
    frequency = np.array([8. ,  8.2,  8.4, 8.6,  8.8,  9. ,  9.2,  9.4,  9.6,  9.8, 10. , 10.2, 10.4, 10.6,10.8, 11. , 11.2, 11.4, 11.6, 11.8, 12. , 12.2, 12.4, 12.6, 12.8, 13. , 13.2, 13.4, 13.6, 13.8, 14. , 14.2, 14.4, 14.6, 14.8, 15. , 15.2, 15.4, 15.6, 15.8])
    phases = np.array([0.        , 1.57079633, 3.14159265, 4.71238898, 0.        , 1.57079633, 3.14159265, 4.71238898, 0.        , 1.57079633, 3.14159265, 4.71238898, 0.        , 1.57079633, 
                       3.14159265, 4.71238898, 0.        , 1.57079633, 3.14159265, 4.71238898, 0.        , 1.57079633, 3.14159265, 4.71238898, 0.        , 1.57079633, 3.14159265, 4.71238898, 
                       0.        , 1.57079633, 3.14159265, 4.71238898, 0.        , 1.57079633, 3.14159265, 4.71238898, 0.        , 1.57079633, 3.14159265, 4.71238898])

    signal_length = 500
    Y = np.array([stimulus(f, phi, signal_length) for f, phi in zip(frequency, phases)]).T
#     Y = np.vstack([referenceSignal(f, 2, 250, 500) for f in np.arange(8, 15.8 + 0.2, 0.2)]).T
#     Y = normalize(Y)

    ca = CCA(n_components = 40, max_iter = 2000)

    base_path = Project.tsinghua_raw_dir / 'benchmark' / 'separated_500'
    outp_path = Project.tsinghua_cca_dir / 'benchmark' / 'separated_500'
    
    ray.shutdown()
    ray.init(_temp_dir='/tmp/something_else')
    
    ray.get([part1.remote(), part2.remote(), part3.remote(), part4.remote()])