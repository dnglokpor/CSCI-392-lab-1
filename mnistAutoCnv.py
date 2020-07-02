# package of custom classes

# imports
from tensorflow.keras import models, layers # for NN configuration and training
from tensorflow.keras.regularizers import l1 # for regularization
from tensorflow.keras import callbacks # for training monitoring
from tensorflow.keras.preprocessing.image import ImageDataGenerator # for data augmentation
from statistics import mean # for computing averages
from math import ceil # for approximation
from sys import exit # for testing purposes only

# info 
class lyrParams:
    '''Class of objects that represents NN layers of any types.
    Expected parameters are position, type, nNeurons, nChannels,
    poolsize, and activation. Configuration depends on layer types:
    '''

    # constructor
    def __init__(self, layerNum, layerCnf):
        '''layerCnf is a list of tuples of attr and val describing a layer'''
        
        self.order = layerNum
        self.params = [val for (attr, val) in layerCnf.items()]
    
    # get specific attribute
    def getOrder(self):
        
        return self.order
    
    # get specific hyperparameter from list
    def getHP(self, n):
        '''return the nth items of self.params'''
        
        return self.params[n]
    
    # describe the layer that can be made from the parameters
    def describeLayer(self):
        '''Describe the layer in an human readable way.'''
        
        
        descr = "{}\t{}\t".format(self.params[0], self.params[1]) # add the layer order (input, hidden, output) and type (conv, dense, etc)
        if self.params[1] == "convolution": 
            descr += "channels: {}\tactivation: {}\tl1 regularization: {}".format(self.params[2], self.params[3],
                                                                                self.params[4])
        elif self.params[1] == "dense":
            descr += "neurons: {}\tactivation: {}\tl1 regularization: {}".format(self.params[2], self.params[3],
                                                                            self.params[4])
        elif self.params[1] == "maxpool":
            descr += "pool size: 2" # it's a fixed value
        elif self.params[1] == "dropout":
            descr += "rate: {}".format(self.params[2])
        else: # nothing else to do
            descr += ""
        
        return descr

class mnistCNN:
    '''Class of CNN that work on the MNIST dataset with a constant input shape of
    (28,28,1). Uses methods from keras.models and keras.layers
    '''  
       
    # constructor
    def __init__(self, nnLayers = None, savedModelFile = None):
        '''Either instantiates an object and uses the passed nnLayers to make an NN model that fits it
        or uses the passed savedModelFile to retrieve the passed model from .h5. In both cases, the created
        object's keras network is not compiled.'''
        
        if nnLayers != None: # making a model from scratch
            self.config = nnLayers # save NN configuration for resetting purposes
            self.nn = models.Sequential() # init the NN
            self.build(self.config) # add the layers

        if savedModelFile != None:
            self.config = list()
            self.nn = models.load_model(savedModelFile, compile = False)

        if nnLayers == savedModelFile == None or (nnLayers != None and savedModelFile != None):
            print("Instances require at least but no more than one of the parameters!")
            exit(1)

    # customs pre setups for possible incoming layers
    def addInput(self, nChannels, activationF):
        '''Supposing all input layers will be convolution networks all we need are number
        of channels and the activation function.'''
        
        self.nn.add(layers.Conv2D(nChannels,(3, 3), activation = activationF, \
            input_shape = (28, 28, 1)))
    
    def addConv(self, nChannels, activationF, regularization):
        '''Add a convolution layer of nChannels channels and activationF activation
        not varying the kernel size.'''
        
        self.nn.add(layers.Conv2D(nChannels,(3, 3), activation = activationF,
            activity_regularizer = l1(regularization)))
    
    def addDense(self, nNeurons, activationF, regularization):
        '''Add a dense layer of nNeurons neurons and activationF activation.'''
        
        self.nn.add(layers.Dense(nNeurons, activation = activationF,
            activity_regularizer = l1(regularization)))
    
    def addMaxpool(self):
        '''Adds a maxpool by poolsize of 2.'''
        
        self.nn.add(layers.MaxPooling2D((2, 2)))

    def addFlatten(self):
        '''Adds a flatenning layer to convert the n-dimensionned data into a 1 dimension.'''
        
        self.nn.add(layers.Flatten())

    def addDropout(self, rate):
        '''Adds a dropout layer by rate.'''
        
        self.nn.add(layers.Dropout(rate))

    def addBatchNorm(self):
        '''Adds a layer that does Batch Normalization of the data.'''
        
        self.nn.add(layers.BatchNormalization())


    # build the layer in conformity to the passed nnLayers
    def build(self, nnLayers):
        '''use the keras.models.add() method to add specific layers from the info provided
        in the nnLayers lyrParams list.
        Expected parameters: [0]is position (always) [1] is type (always)
        if [1] is "convolution"
            [2] is nChannels
            [3] is activation
            [4] is activity regularization via l1(last)
        elif [1] is "dense"
            [2] is nNeurons
            [3] is activation (last)
            [4] is activity regularization via l1(last)
        elif [1] is "dropout"
            [2] is rate (last)
        else [1] is "maxpool" or "flatten" or "batchnorm"
            there are no other parameters
        '''
        
        for layer in nnLayers:
            if layer.getHP(0) == "input": # check for input layer
                self.addInput(layer.getHP(2), layer.getHP(3))
            else:
                if layer.getHP(1) == "convolution": # convolution hidden layer
                    self.addConv(layer.getHP(2), layer.getHP(3), layer.getHP(4))
                elif layer.getHP(1) == "dense": # dense layer
                    self.addDense(layer.getHP(2), layer.getHP(3), layer.getHP(4))
                elif layer.getHP(1) == "maxpool": # maxpool layer
                    self.addMaxpool()
                elif layer.getHP(1) == "flatten": # flatenning layer
                    self.addFlatten()
                elif layer.getHP(1) == "dropout": # dropout layer
                    self.addDropout(layer.getHP(2))
                elif layer.getHP(1) == "batchnorm": # batch normalization layer
                    self.addBatchNorm()
                else:
                    print("No such layer preset!")
                    exit()

    
    # compile the network
    def compileCNN(self):
        '''Simply calls the keras.models.compile with preset parameters for this project.'''
        
        OPT = "rmsprop"
        LOSS = "categorical_crossentropy"
        METRICS = "accuracy"
        
        self.nn.compile(optimizer = OPT, loss = LOSS, metrics = [METRICS])

    # print the summary
    def showNetworkComposition(self):
        '''Show the summary of the built network using keras.models.summary'''
        
        self.nn.summary()
    # resetting NN for multiple training purposes
    def resetCNN(self):
        '''Instantiate a new keras sequential model object build it to match the initial
        configuration.'''
        
        self.nn = models.Sequential()
        self.build(self.config)
    
    # compile, train, test and return results
    def runTest(self, data):
        '''Compile using rms back propagation, categorical cross entropy and accuracy metrics.
        Use the compiled network to train and appreciate on the passed data. data is a tuple of
        [0]training data [1]training labels [2]testing data [3]testing labels.
        All data are the 10k sample of 28*28 grayscale pictures while labels are one-hot vectors of
        dim 10.
        Returns a tuple of [0]best epoch [1]average validation loss [2]average validation accuracy.'''
        
        # make an image modifying generator
        gen = ImageDataGenerator(
            width_shift_range = .6, # 60% possible width shift
            height_shift_range = .6, # 60% possible height shift
            rotation_range = 15, # 15 degree possible rotation range
            shear_range = .6, # 60% possible shearing
        )
        
        # set up accuracy monitor
        # checks when the validation accuracy has stopped increasing and waits 2 epochs before
        pat = 3 # accuracy is unlikely to get back up after a couple of downs
        accMonitor = callbacks.EarlyStopping(monitor = "val_accuracy", patience = pat)
        
        # run tests
        results = list() # save epoch, loss and accuracy for each sessions
        maxNumberEpochs = 15
        bSize = 64
        steps = ((len(data[0]) // bSize) + 1) #* 4 #runs about 4 times the amount of data each epoch
        for session in range(3): # three sessions total
            self.showNetworkComposition() # debug
            
            # compile the network each run through
            self.compileCNN()
            
            # single session of training out of the three
            sessionHist = self.nn.fit(
                gen.flow(data[0], data[1], batch_size = bSize),
                epochs = maxNumberEpochs, # max number we hope to not reach if accuracy drops beforehand
                verbose = 1,
                callbacks = [accMonitor], # catch the loss of accuracy and stops the training
                validation_data = (data[2], data[3]), # validation
                steps_per_epoch = steps
            )
            
            # record session results
            # the best epoch is the epoch with the highest validation accuracy
            sessionEpochs = sessionHist.history["val_accuracy"].index(max(sessionHist.history["val_accuracy"]))
            results.append((sessionEpochs + 1, \
                max(sessionHist.history["val_loss"]), # gets the highes validation loss from data list
                max(sessionHist.history["val_accuracy"]))) # same for validation accuracy
            
            # reset the network before the next training session
            self.resetCNN()

        # print combined results
        print(results)
    
        # return tupple of the three resulting values
        # return shape (best epoch, average loss, average validation)
        return (ceil(mean((results[0][0], results[1][0], results[2][0]))), # ceil of average of epochs
                mean((results[0][1], results[1][1], results[2][1])), # average of losses
                mean((results[0][2], results[1][2], results[2][2])) # average of accuracy
        )
        
    # train a model and save it
    def saveTrainedNN(self, nEpochs, data, savefile):
        '''Build and train the NN model for nEpochs epochs then save the resulting
        model in the file savefile.'''
        
        # make generator
        gen = ImageDataGenerator(
            width_shift_range = .6, # 60% possible width shift
            height_shift_range = .6, # 60% possible height shift
            rotation_range = 15, # 15 degree possible rotation range
            shear_range = .6, # 60% possible shearing
        )
        
        # compile
        self.compileCNN()
        
        # set training parameters
        bSize = 64
        steps = ((len(data[0]) // bSize) + 1) #* 4
        
        # train and validate
        self.showNetworkComposition() # debug
        modelHist = self.nn.fit(
            gen.flow(data[0], data[1], batch_size = bSize),
            epochs = nEpochs, # max number we hope to not reach if accuracy drops beforehand
            verbose = 1,
            validation_data = (data[2], data[3]), # validation
            steps_per_epoch = steps
        )
        
        # saving to h5
        print("saving model as: {}".format(savefile))
        self.nn.save(savefile)
        
    # use a model for prediction
    def classifyMNIST(self, data):
        '''Run a NN model on the passed data to see how model works on the data.
        Returns the n-array of predictions made.
        data is the 10k sample of 28*28 grayscale pictures'''
        
        # compile
        self.compileCNN()
        
        return self.nn.predict(data)