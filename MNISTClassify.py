from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy
from sys import exit

# custom modules
import fileManager as fm # has all the project file names and methods to  automatize required std in and out

# images database processing
(train_data, train_labels), (test_data, test_labels) = mnist.load_data() # loading from keras mnist

# Reshape training and test data to add an additional dimension of 1 channel
test_data = test_data.reshape((10000, 28, 28, 1))

# Revise pixel data to 0.0 to 1.0, 32-bit float
test_data = test_data.astype('float32') / 255

# load the pretrained model
pretrainedModel = fm.cnv.mnistCNN(savedModelFile = fm.MODEL_OUTPUT)
pretrainedModel.showNetworkComposition()
# test and output predictions out of model
predictions = pretrainedModel.classifyMNIST(test_data)

# save it to a file number per number
for prediction in predictions: # for each lists of <numpy.float32>
    highest = 0 # start to the first estimate
    for index in range(1, len(prediction)):
        if prediction[index] > prediction[highest]:
            highest = index # find the index of the max
    # record the index as the value guessed
    fm.predictedOut(highest)
