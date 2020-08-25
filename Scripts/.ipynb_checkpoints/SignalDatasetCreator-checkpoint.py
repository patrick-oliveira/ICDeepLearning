import numpy as np
from utils import generate_time_series

def SignalDatasetCreator(size):
    path = 'C:\\Users\\Patrick\\Documents\\GitHub\\ICDeepLearning\\Data\\Signals'
    frequencies = list(range(10, 20+1))
    for frequency in frequencies:
        new_path = path+'\\'+str(frequency)
        for i in range(500):
            
            perturbation = np.random.normal(0, 1, size = size) # Substitua por uma função adequada
            
            signal = perturbation + generate_time_series(frequency = frequency, size = size)
            np.savetxt(new_path+'\\'+str(i)+'.txt', signal)
            
        for i in range(500, 1000):
            perturbation = np.random.normal(0, 1)
            signal = perturbation + generate_time_series(frequency = frequency, mode = 'cos')
            np.savetxt(new_path+'\\'+str(i)+'.txt', signal)
            