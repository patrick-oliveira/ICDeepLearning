import matplotlib.pyplot as plt
import numpy as np

def createField(method, data, indexes, figSize, figPath):
    """
    Generates and saves into "figPath" a set of pictures using the method "gaf" based on the selected 
    "indexes" from "data".
    
    Inputs:
        method: Gramian Angular (Summation/Difference) field; Markov Transition Field; Recurrence Plot. Class.
        data: Information of the dataset (path, state, subject). Pandas Dataset
        indexes: Subset of data. List.
        figSize: Tuple.
        figPath: String.
    """
    for i in indexes:
        sample = np.loadtxt(data.loc[i]['path']).reshape(1, -1)
        if sample[0][1] != 0:
            X = method.fit_transform(sample)
            
            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = figSize)
            plt.axis('off')
            im = ax.imshow(X[0], cmap = 'rainbow', origin = 'lower')

            state = str(int(data.loc[i]['state']))
            subject = str(data.loc[i]['subject'])
            if data.loc[i]['path'][-6] == '\\':
                file = data.loc[i]['path'][-5]
            else:
                file = data.loc[i]['path'][-6:-4]

            try:
                fig.savefig(figPath+'\\'+state+'\\'+subject+file+'_'+str(i)+".png", dpi = 200, bbox_inches = 'tight')
                plt.close()
            except:
                print("Error creating picture: {}".format(figPath+'\\'+state+'\\0_'+file+".png"))
    