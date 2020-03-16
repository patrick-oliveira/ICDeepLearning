import os
from utils import path
import pandas as pd

def main():
    for filetitle in os.listdir(dataPath()):
    if filetitle[7] == "_":
        os.rename("Dados_BCI/"+filetitle, "Dados_BCI/"+(filetitle[:6] + "0" + filetitle[6:]))

    dataPath = path("Dados_BCI")
    files = os.listdir(dataPath())
    files.sort()

    data = pd.DataFrame(index = range(len(files)), columns = ['ind', 'path', 'state'])
    for i in data.index:
        title = files[i]
        data.loc[i]['ind'] = title[6:8]
        data.loc[i]['path'] = dataPath/title
        data.loc[i]['state'] = int(title[14])

    trainset = data[:150]
    testset  = data[150:]

    data.to_csv("SSVEPDataset.csv", index = False)
    trainset.to_csv("SSVEPTrainset.csv", index = False)
    testset.to_csv("SSVEPTestset.csv", index = False)

#####

main()