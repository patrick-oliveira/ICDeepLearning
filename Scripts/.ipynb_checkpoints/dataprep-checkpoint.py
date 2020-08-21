import os
import pandas as pd
from sklearn.utils import shuffle

def CreateDatasetCSVList(dataType, outputPath = 'Backup\\Originais'):
    dataPath = 'C:\\Users\\Patrick\\Documents\\GitHub\\ICDeepLearning\\Data\\'+outputPath+'\\'+dataType
    subjects = os.listdir(dataPath)
    all_data = []
    
    for subject in subjects:
        subjectPath = dataPath+"\\"+subject
        
        for state in range(1, 5 + 1):
            filesPath = subjectPath+"\\"+str(state) 
            files = os.listdir(filesPath)

            data = pd.DataFrame(index = range(len(files)), columns = ['subject', 'state', 'path'])
            for i in data.index:
                data.loc[i]['subject'] = subject 
                data.loc[i]['path'] = filesPath+"\\"+files[i]
                data.loc[i]['state'] = state
            
            all_data.append(data)

    data = shuffle(pd.concat(all_data))
    data.reset_index(inplace=True, drop=True)
    data.to_csv("SSVEPDataset_"+dataType+".csv", index = False)
    
    
def SplitData(split, dataset, RawDataPath, outDataPath):
    split = {
        '256':1,
        '512':2,
        '768':3,
        '1024':4,
        '1280':5
    }
    Subjects = set()
    Stages = ['1', '2', '3', '4', '5']
    RawDataFiles = os.listdir(RawDataPath)
    # Detecting subjects
    for filename in RawDataFiles:
        Subjects.add(filename[11:24])

    # Creating folders for each subject    
    for subjectName in Subjects:
        os.mkdir(outDataPath+'//'+subjectName)
        for i in Stages:
            os.mkdir(outDataPath+'//'+subjectName+"//"+i)

    # Cut the series based on the specified interval (s), saves it into csvs
    for file in RawDataFiles:
        filename = file[11:24]
        stage = file[-5]
        outputPath = outDataPath+"\\"+filename+"\\"+stage

        data = get_data_from_file(RawDataPath+'\\'+file)
        splittedData = get_splitted_dataset(data[0], split[str(s)])
        k = len(splittedData)

        for i, data in zip(range(1, k), splittedData[:-1]):
            data.to_csv(outputPath+"\\"+str(i)+".csv", sep = ',', header = False, index = False)

        if len(splittedData[-1]) == s:
            splittedData[-1].to_csv(outputPath+"\\"+str(i + 1)+".csv", header = False, index=  False)
    