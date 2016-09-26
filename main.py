from entity import *
from utility import *

dataset = readfile('spambase.data')
splited_data = splitdata(dataset)
training = splited[0]
validation = splited[1]
testing = splited[2]
#print(training[0])


##neur_net = NeuralNet(0, 2, inputs)
##neur_net.train()
##print(neur_net.get_weights())

#n = Neuron(len(inputs[0]), 0.8)

#print(len(inputs[0]))
#print(n.get_weights())
