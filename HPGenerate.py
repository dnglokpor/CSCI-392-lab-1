from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sys import exit

# custom modules
import fileManager as fm # has all the project file names and methods to  automatize required std in and out

# import JSON data
myTests = fm.readConfig()

# images database processing
(train_data, train_labels), (test_data, test_labels) = mnist.load_data() # loading from keras mnist

# Reshape training and test data to add an additional dimension of 1 channel
train_data = train_data.reshape((60000, 28, 28, 1))
test_data = test_data.reshape((10000, 28, 28, 1))

# Revise pixel data to 0.0 to 1.0, 32-bit float
train_data = train_data.astype('float32') / 255
test_data = test_data.astype('float32') / 255

# one-hot encoding of labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# recover the best model configuration from report file
bestModel = fm.recoverBestModel() # returns a (nModel, list of layers, nEpochs, accuracy) tuple

# build the CNN object that matches the model
bestCNN = fm.cnv.mnistCNN(bestModel[1])

# train and save it
bestCNN.saveTrainedNN(bestModel[2], (train_data, train_labels, test_data, test_labels),
                        fm.MODEL_OUTPUT)
