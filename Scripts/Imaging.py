import matplotlib.pyplot as plt
from PIL import Image
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
from typing import Dict

# def createField(method, data, indexes, fig_size, fig_path, signal_length, preprocessing, project, cmap = 'rainbow'):
#     """
#     Generates and saves into "figPath" a set of pictures using the method "gaf" based on the selected 
#     "indexes" from "data".
    
#     Inputs:
#         method: Gramian Angular (Summation/Difference) field; Markov Transition Field; Recurrence Plot. Class.
#         data: Information of the dataset (path, state, subject). Pandas Dataset
#         indexes: Subset of data. List.
#         figSize: Tuple.
#         figPath: String.
#     """
#     for i in indexes:
#         subject = data.loc[i]['subject']
#         state   = data.loc[i]['state']
#         file    = data.loc[i]['file']
#         path = (project.combined_series_dir / preprocessing / signal_length / subject / state / file).__str__()
#         sample = np.loadtxt(path).reshape(1, -1)

#         if sample[0][1] != 0:
#             X = method.fit_transform(sample)
            
#             fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = fig_size)
#             plt.axis('off')
#             im = ax.imshow(X[0], cmap = cmap, origin = 'lower')

#             file = file[:-4]
#             try:
#                 (fig_path / state).mkdir(parents=True, exist_ok=True)
#                 path = (fig_path / state / (subject+'_'+file+'.png')).__str__()
# #                 np.savetxt(path, X[0])
#                 fig.savefig(path, bbox_inches = 'tight')
#                 plt.close()
#             except:
#                 print("Error creating picture: {}".format(path))
                
                
            
# General Classes

class Imaging:
    def __init__(self, path: str = None, pic_args: dict = None, plot: bool = True, savepic: bool = False):
        self.methods = {'gasf': GASF(),
                        'gadf': GADF(),
                        'mtf':  MTF(),
                        'rp':   RP()}
        self.path = path
        self.pic_args = pic_args
        self.plot = plot
        self.savepic = savepic
        
    @property
    def path(self):
        return self._path
    
    @property
    def pic_args(self):
        return self._pic_args
        
    @path.setter
    def path(self, path: str):
        # Modifique isso aqui para construir o endereço com nome do arquivo e diretório
        self._path = path
        
    @pic_args.setter
    def pic_args(self, args: dict = None):
        if args != None:
            assert 'figsize' in args, "\'figsize\' must be provided."
            assert 'cmap' in args, "\'cmap\' must be provided."
            
        self._pic_args = args
        
    def apply(self, method: str, X, **kwargs):
        pic_matrix = self.methods[method].apply(X, **kwargs)
        
        if self.plot:
            self.plot_result(pic_matrix)
            
        return pic_matrix
    
    def plot_result(self, result):
        figsize = self.pic_args['figsize'] if self.pic_args != None and 'figsize' in self.pic_args else (12, 7)
        cmap    = self.pic_args['cmap'] if self.pic_args != None and 'cmap' in self.pic_args else 'Greys'

        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = figsize)
        plt.axis('off')
        im = ax.imshow(result[0], cmap = cmap, origin = 'lower')
        plt.show()

        if self.savepic:
            try:
                path = self.path if self.path != None else 'Imaging_Result.png'
                fig.savefig(path, bbox_inches = 'tight')
            except:
                print(f"Error saving picture: {self.path}")

class ImagingMethod:
    def __init__(self):
        pass
    
    def apply(self, X, **kwargs):
        return self._method(X, **kwargs) 
    
    def _method(self, **kwargs):
        raise NotImplementedError
        
        
# Specific Classes
        
class GADF(ImagingMethod):
    def _method(self, X, **kwargs):
        gadf = GramianAngularField(method = 'difference', **kwargs)
        
        return gadf.fit_transform(X)
    
class GASF(ImagingMethod):
    def _method(self, X, **kwargs):
        gasf = GramianAngularField(method = 'summation', **kwargs)
        
        return gasf.fit_transform(X)
    
class MTF(ImagingMethod):
    def _method(self, X, **kwargs):
        mtk = MarkovTransitionField(**kwargs)
        
        return mtk.fit_transform(X)
    
class RP(ImagingMethod):
    def _method(self, X, **kwargs):
        rp = RecurrencePlot(**kwargs)
        
        return rp.fit_transform(X)
        
        