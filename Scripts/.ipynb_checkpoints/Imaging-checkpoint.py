import matplotlib.pyplot as plt
from PIL import Image
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
from typing import Dict, List
from multiprocessing import Pool
from path import Path
from itertools import islice, product, chain
from functools import partial
        
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
            self.save_pic()
            
    def save_pic(self):
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
        
# Creating a dataset of pictures

def split_list(input_list: List, number_of_slices: int) -> List[List]:
    '''
    Create a list of slices from 'input_list' of fixed size.
    
    If L = len(input_list) and n = number_of_slices, then the Division Algorithm says that there are integers q and r < n such that L = q*n + r. If L is divisible by n, then the output list will be a set of n slices of size q, otherwise, there will be r slices of size q + 1 and and n - r slices of size q.
    
    Args:
        input_list (List): List to be segmented into sublists of potentially equal size.
        segment_size (Int): Number of slices.
        
    Output:
        output_segments (List[List]): A list of slices of fixed size 
    '''
    L = len(input_list)
    n = number_of_slices
    input_list = iter(input_list)
    
    if L// n != 0:
        split_groups_sizes = [L//n]*n
        for i in range(L%n):
            split_groups_sizes[i] = split_groups_sizes[i] + 1
    else:  
        split_groups_sizes = [L]
    
    output_segments = [list(islice(input_list, size)) for size in split_groups_sizes]
    
    return output_segments if len(output_segments) != 1 else output_segments[0]

# def transform_series_from_folder(files_path: List[str], output_paths: List[str], interface: Imaging, pic_args: Dict = None, workers: int = 1, **kwargs):
#     '''
    
#     Args:
#         files_path (List[str]):
#         output_paths (List[str]):
#         interface (Imaging):
#         pic_args (Dict):
#         workers (int):
#     '''
#     assert len(files_path) == len(output_path), 'For each list of files there must be an output path.'
    
#     files_list = [os.listdir(Path(p)) for p in files_path]
#     files_with_address_list = [product(files_list[k], [output_path[k]], [files_path[k]]) for k in range(len(output_path))]
#     files_with_address_list = [*chain.from_iterable(files_with_address_list)]
#     splitted_files_with_address_list = split_list(files_with_address_list, num_workers)
    
#     interface.savepic  = True
#     interface.plot     = False
#     interface.pic_args = pic_args
    
#     f = partial(_transform_series_from_folder, P_output, interface)
#     with p as Pool(num_workers):
#         p.map(f, files_list)
            
# def _transform_series_from_folder(interface: Imaging, files_list: List[(str, str)])
#     for file, output_path, input_path in files_list:
#         X = np.loadtxt(f"{input_path}/{file}")
#         interface.path = f"{output_path}/{re.sub('[.].*', '.png', file)}"
#         Imaging.apply(X)
        
        
        

        