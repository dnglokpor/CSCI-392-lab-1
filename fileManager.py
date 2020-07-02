# python json loader

import json
from json import JSONEncoder
import logging
from sys import exit

#custom modules
import mnistAutoCnv as cnv # contains objects and methods that allows keras models training required by the project

# constant filenames
NETWORKS_CONFIG_FILE = "layersconfig.json"
REPORT_FILE = "HPTest.out"
MODEL_OUTPUT = "MNIST.h5"
PREDICT_OUTPUT = "MNISTClassify.out"

# parsing the data
def readConfig():
    '''Load the contents of the JSON external file into the program as a list of 
    lyrParams.'''
    
    # reading info from json file
    try:
        ioData = open(NETWORKS_CONFIG_FILE,)
    except FileNotFoundError:
        print("Path to datafile compromised.")
        exit(1)
    else:
        try:
            contents = json.load(ioData)
        except json.decoder.JSONDecodeError:
            print("No data in datafile or JSON encoding erroneous.")
            exit(1)
        else:
            testsNNs = list() # all the tests
            testsNames = list() # all their names
            for test in contents["tests"]: # iterate through each test in the json
                for testName, testNN in test.items(): # read test names and NN configs
                    testsNames.append(testName) # save the names for reference
                    # get the info of each layer and save them as a NN item
                    NN = [cnv.lyrParams(layerNum, layerConfig) for layerNum, layerConfig in enumerate(testNN)]
                    testsNNs.append(NN) # save all NN items
            return testsNNs
    
    # close the file after reading
    ioData.close()

# write report in a file
def resultsReport(report):
    ''' uses the logging module to quickly save the report in a file'''
    
    logging.basicConfig(filename = REPORT_FILE,
                        format = "%(message)s",
                        filemode = "w")
                        
    # making log object
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    
    # log the report
    logger.info(report)

# recover the best setup
def recoverBestModel():
    '''Using read config, it remembers the config of the networks and use it to
    navigate through the report file to and check each report result. Records
    the highest validation accuracy and returns it alongside the related epoch.'''
    
    configs = readConfig() # recover the config
    
    #open the report file
    try:
        ioData = open(REPORT_FILE,'r')
    except FileNotFoundError:
        print("Path to datafile compromised.")
        exit(1)
    else:
        # the file is open. we can navigate it
        no = 0
        acc = 0
        nEpochs = 0
        nnConfig = list()
        for i, config in enumerate(configs): # each config (list of lyrParams)
            ioData.readline() # skip report title ">>>test i<<<"
            for layer in range(len(config)): # for each layer in the config
                ioData.readline() # skip each layer description
            # now we are at the beginning of the line with the info
            resultsLineWords = ioData.readline().split() # load line in memory as list of words
            # the line split looks like this:
            # ["Best", "validation", "at", "epoch", "nEpochs", "with", "loss", "val_loss", "and",
            #   "accuracy", "val_acc"]
            # data is located at: [4]number of epochs [7]validation loss [10]
            if acc < float(resultsLineWords[10]):
                no = i + 1
                nnConfig = config # saves the network configuration for reproduction
                acc = float(resultsLineWords[10]) # save current accuracy
                nEpochs = int(resultsLineWords[4]) # and number of epochs
            ioData.readline() # skip the empty line that separates two reports
        
        return (no, nnConfig, nEpochs, acc)
    # close the file after reading
    ioData.close()

# write predictions
def predictedOut(number):
    ''' uses the logging module to quickly save the predictions to a file.'''
    
    logging.basicConfig(filename = PREDICT_OUTPUT,
                        format = "%(message)s",
                        filemode = "w")
                        
    # making log object
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    
    # log the report
    logger.info("{}".format(number))

if __name__ == "__main__":
    # tests of module methods
    
    # readConfig()
    '''myTests = readConfig()
    for i, NN in enumerate(myTests):
        print("test{}>".format(i))
        for num, config in enumerate(NN):
            print("layer {}> {}".format(config.order + 1, config.params))
        print()
    '''
    
    # recoverBestModel()
    '''best = recoverBestModel()
    print("\nBest CNN model is {}:".format(best[0]))
    for layer in best[1]:
        print(layer.describeLayer())
    print("{} accuracy at {} epochs".format(best[3], best[2]))
    '''
