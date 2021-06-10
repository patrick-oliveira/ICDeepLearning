import scipy.io                      # para ler os arquivos .mat originais
import os                            # gerenciar pastas e arquivos
import numpy as np
from ray.util.multiprocessing import Pool     # processamento em paralelo
from functools import partial        # aplicacao parcial de funcoes
from pathlib import Path             # gerenciar enderecos das pastas e arquivos
from typing import List              # indicar os tipos nas funcoes

def extractSeparatedFiles(dataset:str, raw_data_directory:str = 'D:\\Datasets\\Tsinghua'):
    '''
    Extract data from each trial (40, one for each class) and blocks (one for each trial) and save it separately,
    divided by class/trial and identified by individual and block: S{individual}_{block}.csv
    '''
    raw_data_directory = Path(raw_data_directory + '\\' + dataset)
    files = [file for file in os.listdir(raw_data_directory) if "mat" in file]
    files = segmentList(files, 4)
    
    for trial in range(40):
        (raw_data_directory / 'separated' / str(trial + 1)).mkdir(parents = True, exist_ok = True)
    
    if dataset == 'benchmark':
        B = 6
    else:
        B = 4
        
    f = partial(_extractSeparatedFiles, dataset = dataset, 
                                        raw_data_directory = raw_data_directory, 
                                        num_blocks = B)
    
    with Pool(processes = 4) as pool:
        pool.map(f, files)
        
def _extractSeparatedFiles(files: List[str], dataset:str, raw_data_directory:str, num_blocks:int):
    for file in files:
        data = getData(raw_data_directory / file, dataset)
        subject = file.replace('.mat', '')
        for block in range(num_blocks):
            for trial in range(40):
                X = data[:, :, trial, block].T
                output_path = raw_data_directory / 'separated' / str(trial + 1)
                np.savetxt(output_path / 'S{}_{}.csv'.format(subject, block + 1), X)

def getData(path, dataset:str):
    if dataset == 'benchmark':
        subject_data = scipy.io.loadmat(path)['data']
    else:
        subject_data = scipy.io.loadmat(path)['data'][0][0][0].transpose((0, 1, 3, 2))
    return subject_data

def segmentList(L: List, n: int) -> List[List]:
    step = int(len(L)/n)
    assert step >= 2, 'floor[len(L)/n] must be >= 2' 
    
    segments = []
    for k in range(0, len(L) + 1, step):
        segments.append(L[k:k+step])
        
    if len(segments[-1]) < step:
        segments[-2].extend(segments[-1])
        segments = segments[:-1]
        
    return segments

def splitDataset(dataset:str, outputPath:str, time:int, inputPath:str = 'D://Datasets//Tsinghua'):
    if dataset == 'benchmark':
        assert 1500%(time*250) == 0
        num_blocks = 6
    else: 
        assert 750%(time*250) == 0
        num_blocks = 4
        
    datasetPath = Path(inputPath + '//{}//separated'.format(dataset))
    outputPath = Path(outputPath + '//{}//{}'.format(dataset, str(time*250)))
    outputPath.mkdir(parents = True, exist_ok = True)
    sublists = [list(range(x, y)) for x, y in [(1, 11), (11, 21), (21, 31), (31, 41)]]
    
    f = partial(_splitDataset, inputPath = datasetPath, outputPath = outputPath, time = time)
    with Pool(processes = 4) as pool:
        pool.map(f, sublists)
    
def _splitDataset(classes: List[int], inputPath, outputPath, time:int):
    for c in classes:
        input_path = inputPath / str(c)
        output_path = outputPath / str(c)
        output_path.mkdir(parents = True, exist_ok = True)
        files = os.listdir(input_path)
        for file in files:
            X = np.loadtxt((input_path / file).__str__())
            
            try:
                n_segments = int(X.shape[0]/(time*250))
                segments = np.vsplit(X, n_segments)
            except:
                print((input_path / file).__str__())
                print(n_segments)
                
            for k in range(len(segments)):
                np.savetxt(output_path / file.replace('.csv', '_{}.csv'.format(k)), segments[k])
    