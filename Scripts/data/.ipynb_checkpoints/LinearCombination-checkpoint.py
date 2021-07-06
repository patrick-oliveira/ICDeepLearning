import os
import pandas as pd
from sklearn.utils import shuffle
import numpy as np

def CreateDatasetCSVList(signal_length, data_path, output_path):
    """
    Creates a csv file to hold information of the original data, i.e. signals 
    data_type: int
    data_path: Path - Path for the original dataset.
    output_path: Path - Path for 
    """

    subjects = os.listdir(data_path)
    all_data = []
    
    for subject in subjects:
        subject_path = data_path / subject
        
        for state in range(1, 5 + 1):
            files_path = subject_path / str(state)
            files = os.listdir(files_path)
            # coloque uma função para remover tudo o que não for csv

            data = pd.DataFrame(index = range(len(files)), columns = ['subject', 'state', 'file'])
            
            for i in data.index:
                data.loc[i]['subject'] = subject
                data.loc[i]['state'] = state
                data.loc[i]['file'] = files[i]
                
            all_data.append(data)

    data = shuffle(pd.concat(all_data))
    data.reset_index(inplace=True, drop=True)
    data.to_csv(output_path / ("SSVEPDataset_"+signal_length+".csv"), index = False)

def read_data(path):
    return pd.read_csv(path, header = None, dtype = np.float64)


electrodes_dict = {'O1':1, 'O2':2, 'Oz':3, 'POz':4, 'Pz':5, 'PO3':6, 'PO4':7, 'PO7':8, 
                   'PO8':9, 'P1':10, 'P2':11, 'Cz':12, 'C1':13, 'C2':14, 'CPz':15, 'FCz':16}

def combine(project, signal_length, preprocessing = 'CCA', 
            electrodes = ['O1', 'O2', 'Oz', 'POz', 'Pz', 'PO3', 'PO4', 'PO7', 'PO8', 'P1', 'P2', 'Cz', 'C1', 'C2', 'CPz', 'FCz'],
            ):
    """
    project : Project
    data_type : String

    """
    CreateDatasetCSVList(signal_length, 
                         project.unicamp_signals / preprocessing / signal_length,
                         project.output)

    data = pd.read_csv(project.output / ("SSVEPDataset_"+signal_length+".csv"), dtype = 'str')

    for i in range(len(data)):
        subject = data.loc[i]['subject']
        state   = data.loc[i]['state']
        file    = data.loc[i]['file']
        sample_path = project.unicamp_signals / preprocessing / signal_length / subject / state / file
        sample = read_data(sample_path).values.T

        num_electrodes = len(electrodes)
        combined_series = np.zeros(int(signal_length))
        for electrode in electrodes:
            combined_series += sample[electrodes_dict[electrode] - 1]
        combined_series /= num_electrodes

        (project.unicamp_combined_signals / preprocessing / signal_length / subject / state).mkdir(parents=True, exist_ok=True)
        combined_path = project.unicamp_combined_signals / preprocessing / signal_length / subject / state / file
        np.savetxt(combined_path.__str__(), combined_series)