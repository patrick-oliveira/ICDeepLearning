import os
from utils import path
import pandas as pd
from sklearn.utils import shuffle

def main(dataType):
#     dataPath = path('/home/roboto/Documents/GitHub/ICDeepLearning/Data/Originais/768')
    dataPath = path('C:\\Users\\Patrick\\Documents\\GitHub\\ICDeepLearning\\Data\\Originais\\'+dataType)
    n_data = len(os.listdir(dataPath()))
    all_data = []
    
    for dataset in range(1, n_data+1):
        pstr = dataPath/str(dataset)
        for filetitle in os.listdir(pstr):
            if filetitle[7] == "_":
                os.rename(pstr+'/'+filetitle, pstr+'/'+(filetitle[:6] + "0" + filetitle[6:]))

        files = os.listdir(pstr)
        files.sort()

        data = pd.DataFrame(index = range(len(files)), columns = ['ind', 'path', 'state'])
        for i in data.index:
            title = files[i]
            data.loc[i]['ind'] = title[6:8]
            data.loc[i]['path'] = (pstr+"/"+title)[-21:]
            data.loc[i]['state'] = int(title[14])
            
        all_data.append(data)

    data = shuffle(pd.concat(all_data))
    data.reset_index(inplace=True, drop=True)
    testsetSize = int(len(data)*0.25)
    testset = data[:testsetSize]
    trainset  = data[testsetSize:]

    data.to_csv("SSVEPDataset.csv", index = False)
    trainset.to_csv("SSVEPTrainset.csv", index = False)
    testset.to_csv("SSVEPTestset.csv", index = False)

#####

main('512')