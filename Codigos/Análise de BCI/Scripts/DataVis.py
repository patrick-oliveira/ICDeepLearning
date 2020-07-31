def batchVisualization(series = None, dataloader = None):
    if(dataloader != None):
        sample = iter(dataloader).next()
        series = sample['series'][0].squeeze(0)
    
    fig, ax = plt.subplots(nrows = 16, ncols = 1, figsize = (10, 50))

    for i, axis in enumerate(ax):
        axis.plot(range(512), series[:, i], color = 'k')
        axis.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0))
        
def createInputFigure(inp):
    """
    Creates a 512x512 matrix.
    input:
        input: a dictionary; {"series": 4 tensors, "class": list of size 4}
    """
    def increaseSize(X):
        output = X
        for i in range(31):
            output = torch.cat((output, X), 0)
        return output
    
    return { "series": torch.stack([increaseSize(X) for X in inp['series']]),
             "class": inp['class'] }