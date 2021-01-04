import pandas as pd
import os


#Retorna a classe do dataset baseado em (índice do valor +1) + 1
def get_class(header):
    return header[header > 0].apply(lambda x: x.index.get_loc(x.first_valid_index()), axis=1)[0] + 1

#Retorna um dataset de arquivo
def get_data_from_file(file, cols='all'):
    pds = pd.read_csv(file, sep=" ", header=None, skiprows=1)
    pds.columns = ['O1', 'O2', 'Oz', 'POz', 'Pz', 'PO3', 'PO4', 'PO7', 'PO8', 'P1', 'P2', 'Cz', 'C1', 'C2', 'CPz', 'FCz']
    header = pd.read_csv(file, sep=" ", header=None, nrows=1)
    lbl = get_class(header)
    
    if cols != 'all':
        return pds[cols], lbl
    return pds, lbl

# Retorna dataframe em segmentos de n_segundos (256Hz * n_seconds) 
def get_splitted_dataset(df, n_seconds=1):
    return [df.loc[i:i+(int(256 * n_seconds))-1,:] for i in range(0, len(df),(int(256*n_seconds)))]


def load(series_length, raw_data_path, out_data_path):
    """
    series_length : String
    raw_data_path : Path
    out_data_path : Path
    """
    split = {
        '128':0.5,
        '256':1,
        '512':2,
        '768':3,
        '1024':4,
        '1280':5
    }

    subjects = set()
    stages = ['1', '2', '3', '4', '5']
    raw_data_files = os.listdir(raw_data_path)

    # Identifying subjects
    for filename in raw_data_files:
        subjects.add(filename[11:24])

    # Creating folders for each subject    
    for subject_name in subjects:
        (out_data_path / subject_name).mkdir(parents=True, exist_ok=True)
        for i in stages:
            (out_data_path / subject_name / i).mkdir(exist_ok = True)

    # Cut the series based on the specified interval (s), saves it into csvs
    for file in raw_data_files:
        filename = file[11:24] # extract the subject name
        stage = file[-5] # extract the stage
        output_path = out_data_path / filename / stage

        data = get_data_from_file(raw_data_path / file)
        splitted_data = get_splitted_dataset(data[0], split[series_length])
        k = len(splitted_data)

        # saves the splitted series
        for i, data in zip(range(1, k), splitted_data[:-1]):
            data.to_csv(output_path / (str(i) + '.csv'), sep = ',', header = False, index = False)

        # verify if the last split has the specified length
        if len(splitted_data[-1]) == split:
            splitted_data[-1].to_csv(output_path / str(i + 1) / '.csv', sep = ',', header = False, index = False)