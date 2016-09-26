import math
from definition import *

alpha = 0.8
a = 1
b = 0.5
w = [0, 0, 0, 0]
x1 = [0.8, 0.5, 0.0]
x2 = [0.9, 0.7, 0.3]
x3 = [1.0, 0.8, 0.5]
x4 = [0.0, 0.2, 0.3]
x5 = [0.2, 0.1, 1.3]
x6 = [0.2, 0.7, 0.8]
training_pairs = [[x1, 0], [x2, 0], [x3, 0], [x4, 1], [x5, 1], [x6, 1]]

def activate_derv(synaptic_in):
    return activate(synaptic_in) * (1 - activate(synaptic_in))

neurnet = NeuralNet(2, 3)

converged = False
epoch = 0
emax = 1e-3
eps = 1e-5
count = 5000

print(neurnet.train(training_pairs))

##while(count > 0 and not converged):
##    err = 0
##        
##    for pair in training_pairs:
##        output = output_layer.train(pair)
##        output_layer.adjust_weights()
##        err += output_layer.getLayerError()
##
##    
##          
##    if abs(err - emax) < eps:
##        converged = True
##        
##    epoch += 1
##    count -= 1
##else:
##    print(err)
##    print(output)
    

print("{}{}".format("ANN converges after Epoch ", epoch))
print("{}{}".format("Final Weights: ", neurnet.get_weights()))
    
