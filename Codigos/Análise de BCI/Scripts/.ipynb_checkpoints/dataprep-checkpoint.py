import os
from utils import path
import pandas as pd
from sklearn.utils import shuffle

def dataprep(dataType, outputPath = 'Backup\\Originais'):
#     dataPath = path('/home/roboto/Documents/GitHub/ICDeepLearning/Data/Originais/768')
    dataPath = path('C:\\Users\\Patrick\\Documents\\GitHub\\ICDeepLearning\\Data\\'+outputPath+'\\'+dataType)
    subjects = os.listdir(dataPath())
#     n_data = len(subjects)
    all_data = []
    
    for subject in subjects:
        subjectPath = path(dataPath()+"\\"+subject)
        
        for state in range(1, 5 + 1):
            filesPath = path(subjectPath()+"\\"+str(state)) 

            files = os.listdir(filesPath())

            data = pd.DataFrame(index = range(len(files)), columns = ['subject', 'state', 'path'])
            for i in data.index:
                data.loc[i]['subject'] = subject 
                data.loc[i]['path'] = filesPath()+"\\"+files[i]
                data.loc[i]['state'] = state
            
            all_data.append(data)

    data = shuffle(pd.concat(all_data))
    data.reset_index(inplace=True, drop=True)
#     testsetSize = int(len(data)*0.25)
#     testset = data[:testsetSize]
#     trainset  = data[testsetSize:]

    data.to_csv("SSVEPDataset_"+dataType+".csv", index = False)
#     trainset.to_csv("SSVEPTrainset.csv", index = False)
#     testset.to_csv("SSVEPTestset.csv", index = False)

#####