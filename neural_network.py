from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import SoftmaxLayer
from pybrain.tools.customxml.networkreader import NetworkReader
from pybrain.tools.customxml.networkwriter import NetworkWriter
import numpy
import os
import matplotlib.pyplot as plot

class MyNeuralNetwork:
    def __init__(self, numInputLayers, numHiddenLayers, numOutputLayers): # TODO: initialize weights from file
        if os.path.isfile('facedetect.xml'):
            print "reading from file..."
            self.neuralNetwork = NetworkReader.readFrom('facedetect.xml')
            print "done reading from file..."
        else:
            self.neuralNetwork = FeedForwardNetwork()
            inputLayer = LinearLayer(numInputLayers)
            hiddenLayer = SigmoidLayer(numHiddenLayers)
            outputLayer = SoftmaxLayer(numOutputLayers)

            self.neuralNetwork.addInputModule(inputLayer)
            self.neuralNetwork.addModule(hiddenLayer)
            self.neuralNetwork.addOutputModule(outputLayer)

            inputToHidden = FullConnection(inputLayer, hiddenLayer)
            hiddenToOut = FullConnection(hiddenLayer, outputLayer)

            self.neuralNetwork.addConnection(inputToHidden)
            self.neuralNetwork.addConnection(hiddenToOut)

            self.neuralNetwork.sortModules()

    def activate(self, inputs):
        return self.analyzeResult(self.neuralNetwork.activate(inputs))

    def analyzeResult(self, results):
        max_result = results[0]
        max_index = 0

        for i in range(1, len(results)):
            if results[i] > max_result:
                max_result = results[i]
                max_index = i

        return max_index

    def train(self, x_train, y_train, input_size, class_labels):
        print ("input size = %d" % (input_size))
        ds = ClassificationDataSet(input_size, nb_classes=len(class_labels))

        for i in range(len(x_train)):
            ds.appendLinked(x_train[i], [y_train[i]])

        ds._convertToOneOfMany(bounds=[0, 1])
        trndata, tstdata = ds.splitWithProportion(0.25)

        trainer = BackpropTrainer(self.neuralNetwork, trndata, learningrate=0.01, momentum=0.9)
        trnerror, valerror = trainer.trainUntilConvergence(dataset=trndata, maxEpochs=1000)
        #print trainer.totalepochs
        #plot.plot(trnerror, 'b', valerror, 'r')
        #plot.show()

        out = self.neuralNetwork.activateOnDataset(tstdata).argmax(axis=1)
        print percentError(out, tstdata['target'])
        NetworkWriter.writeToFile(self.neuralNetwork, 'facedetect.xml')
