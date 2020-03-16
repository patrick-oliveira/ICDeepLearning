class NeuralNetwork():
    def __init__(self, neuronsPerLayer):
        """
        Constructor.
        input:
            neuronsPerLayer:list := A list of integers [n0, n1, ..., nr]. For all i = 1, ..., r - 1, ni is
            the number of neurons at layer i; n0 is the input size, and nr the size of the output layer.
        """
        self.network = self.initialize_network(neuronsPerLayer)
        
    def initialize_network(self, neuronsPerLayer):
        """
        Initialize the network, calling the method that will generate the weight matrices for all layers.
        This method should be adapted to use the Strategy Pattern after implementing other types of layers.
        
        
        input:
            neuronsPerLayer:list := A list of integers [n0, n1, ..., nr]. For all i = 1, ..., r - 1, ni is
            the number of neurons at layer i; n0 is the input size, and nr the size of the output layer.
        """
        network = []
        
        for l in range(1, len(neuronsPerLayer)):
            layer = self.linearLayer(neuronsPerLayer[l - 1], neuronsPerLayer[l])
            network.append(layer)
            
        return network
    
    def linearLayer(self, inputSize, neuronsNumber):
        """
        Generates a layer of perceptrons.
        
        input:
            inputSize:int := the number of neurons of the previous later, not counting the bias.
            neuronsNumber:int := number of neurons at the currect layer.
        """
        return {'weights' : np.random.randn(inputSize + 1, neuronsNumber)}
    
    def fit(self, data, target, learningRate = 0.1, epochs = 5, normalize = True, appendOnes = True):
        """
        Train the network, given a training dataset.
        This method should work with a training set and a validation set. The loss function should be computed
        using both sets. 
        input:
            data:numpy array := An nxm array, where n is the sample size, m is the 
            number of features.
                           [x11 ... x1m]
                    data = [... ... ...]
                           [xn1 ... xnm]
                  It is assumed that m equals the size of the input layer.
                  
            target:numpy array := An nx1 array, where n is the sample size.
            
            learningRate:float := Learning rate.
            
            epochs:int := how many times the algorithm should pass through the data
        """
        if normalize == True:
            data = self.normalize(data)
        if appendOnes == True:
            data = self.appendOnes(data)
            
        self.learningRate = learningRate
        self.epochs = epochs
        
        index = np.arange(data.shape[0])
        Loss = []
        for epoch in range(epochs):
            np.random.shuffle(index)
            for i in index:
                X = data[i]
                y = target[i]
                
                o = self.forward(X)
                self.backward(y, o)
            
            Loss.append(self.Loss(data, target))
            
        return Loss
    
    def forward(self, X):
        """
        Forward propagation.
            Each propagation saves, for each layer, its input, local field and output,
            to be used later with the backpropagation algorithm.
        
        input:
            X: A row vector, and x[0] = bias.
        """
        Y = X
        for layer in self.network:
            layer['input'] = Y
            layer['localField'] = self.localField(layer['weights'], Y)
            layer['output'] = self.sigmoid(layer['localField']) 
            Y = self.appendOnes(layer['output'])
        
        return layer['output']
    
    def backward(self, target, output):
        """
        Not a very elegant code. It can be improved.
        First, the error is propagated backwards and the deltas are computed. The weights are corrected afterwards.
        input:
            target: a row vector. The desired value.
            output:  a row vector. The network's output.
        """
        # last layer
        layer = self.network[-1]
        error = target - output
        deltaSignal = error*self.derivativeSigmoid(layer['output'])
        layer['delta'] = deltaSignal
        
        # error propagation
        n_layers = len(self.network)
        for layer in range(n_layers - 2, -1, -1): # The first and last layers are not considered
            nextLayer = self.network[layer + 1]
            currentLayer = self.network[layer]
            # This try/catch is here because the dimensions are not always correct.
            # It's a provisional solution.
            try:
                error = nextLayer['delta'] @ nextLayer['weights'].T
            except:
                error = self.correctDimension(nextLayer['delta']) @ nextLayer['weights'].T
            currentLayer['delta'] = self.derivativeSigmoid(currentLayer['output'])*error[1:]
        
        # Corrections
        for layer in self.network:
            layer['weights'] += self.learningRate*self.correctDimension(layer['input'])@self.correctDimension(layer['delta']).T

        
    def localField(self, weights, inputs):
        """
        Assumes that the weights are column vectors and the inputs are row vectors. It is assumed also
        that inputs[0] = bias.
        """
        return inputs.dot(weights)
    
    def appendOnes(self, X):
        """
        Append a vector of 1s to X.
        inputs:
            X: An nxm array, where n is the sample size, m is the number of features.
        """
        if(len(X.shape) == 1):
            return np.concatenate((np.ones(1), X))
        else:
            ones = np.ones((X.shape[0], 1))
            return np.concatenate((ones, X), axis = 1)
            
    def correctDimension(self, X):
        if(len(X.shape) == 0):
            return np.asarray([X])
        else:
            return X.reshape((X.shape[0], 1))   

    def normalize(self, X):
        return (X - np.mean(X))/np.std(X)
    
    def sigmoid(self, v):
        return 1 / (1 + np.exp(-v))
    
    def derivativeSigmoid(self, y):
        return y*(1 - y)
    
    def tanh(self, x):
        return (1.0 - np.exp(-2*x))/(1.0 + np.exp(-2*x))
    
    def derivativeTanh(self, x):
        return (1.0 + self.tanh(x))*(1 - self.tanh(x))

    def Loss(self, X, y):
        return ((y - self.forward(X)).T.dot(y - self.forward(X))).item()
    
    def getArchitecture(self):
        return self.network
    
    def predict(self, X, normalize = True, appendOnes = True):
        if normalize == True:
            X = self.normalize(X)
        if appendOnes == True:
            X = self.appendOnes(X)
        return np.round(self.forward(X))
    