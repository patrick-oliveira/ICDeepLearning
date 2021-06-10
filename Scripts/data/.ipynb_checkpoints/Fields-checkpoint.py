import numpy as np
import matplotlib.pyplot as plt

def createField(method, data, indexes, fig_size, fig_path, signal_length, preprocessing, project, cmap = 'rainbow'):
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
        subject = data.loc[i]['subject']
        state   = data.loc[i]['state']
        file    = data.loc[i]['file']
        path = (project.combined_series_dir / preprocessing / signal_length / subject / state / file).__str__()
        sample = np.loadtxt(path).reshape(1, -1)

        if sample[0][1] != 0:
            X = method.fit_transform(sample)
            
            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = fig_size)
            plt.axis('off')
            im = ax.imshow(X[0], cmap = cmap, origin = 'lower')

            file = file[:-4]
            try:
                (fig_path / state).mkdir(parents=True, exist_ok=True)
                path = (fig_path / state / (subject+'_'+file+'.png')).__str__()
#                 np.savetxt(path, X[0])
                fig.savefig(path, bbox_inches = 'tight')
                plt.close()
            except:
                print("Error creating picture: {}".format(path))
    