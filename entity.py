import pdb
class Neuron:
    'Definition of a single neuron'
    def __init__(self, num_inputs, learning_rate):
        import random
        self.weights = []
        for num in range(num_inputs):
            self.weights.append(random.uniform(-10, 10))
        self.learning_rate = learning_rate
        self.error = 0
        self.synaptic_in = 0
        self.input_pair = []
        self.delta = 0

    def train(self, input_pair):
        self.input_pair = input_pair
        self.synaptic_in = sum([wgt * ipt for wgt, ipt in zip(self.weights, self.input_pair[0])])
        output = self.activate()
        self.error = self.calculate_err(output)
        return output

    def calculate_err(self, output):
##        if (self.inputs[1] == 0 and output - 1 / 2 > 1e-5) or (self.inputs[1] == 1 and output - 1 / 2 <= 1e-5):
##            return 1
##        return 0

        return self.input_pair[1] - output

    def adjust_weights_def(self):
        self.delta = self.error * self.activate_derv()
        self.weights = [wgt + self.learning_rate * self.delta * ipt for wgt, ipt in zip(self.weights, self.input_pair[0])]

    def adjust_weights(self, delta):
        self.delta = delta * self.activate_derv()
        self.weights = [wgt + self.learning_rate * self.delta * ipt for wgt, ipt in zip(self.weights, self.input_pair[0])]

    def get_weights(self):
        return self.weights

    def activate(self):
        import math
        return 1 / (1 + math.exp(-self.synaptic_in))

    def activate_derv(self):
        return self.activate() * (1 - self.activate())
    
    def setAlpha(self, new_alpha):
        self.alpha = new_alpha

    def getError(self):
        return self.error

    def getDelta(self):
        return self.delta

class Layer:
    'Consisting of some Neurons'
    alpha = 0.8

    def __init__(self, num_neur, num_inputs):
        
        self.neurons = []
        self.layer_error = 0
        self.num_inputs = num_inputs
        for num in range(num_neur):
            self.neurons.append(self.createNeuron(num_inputs + 1))

    def train(self, input_pair):
        errors = []
        output = []
        input_cpy = [input_pair[0] + [1]] + [input_pair[1]]
        
        for neur in self.neurons:
            output.append(neur.train(input_cpy))
            errors.append(neur.getError())
        self.layer_error = sum([error ** 2 for error in errors]) / 2
        return [output, input_cpy[1]]

    def adjust_weights_def(self):
        deltas = []
        for neur in self.neurons:
            neur.adjust_weights_def()
            deltas.append(neur.getDelta())
        return self.next_deltas(deltas)
        
    def adjust_weights(self, deltas):
        for neur, delta in zip(self.neurons, deltas):
            neur.adjust_weights(delta)
        deltas = []
        for neur in self.neurons:
            deltas.append(neur.getDelta())
        return self.next_deltas(deltas)

    def createNeuron(self, num_inputs):
        return Neuron(num_inputs, Layer.alpha)

    def getLayerError(self):
        return self.layer_error

    def get_weights(self):
        return [n.get_weights() for n in self.neurons]

    def next_deltas(self, deltas):
        new_deltas = []
        for num in range(self.num_inputs):
            weights = []
            for neur in self.neurons:
                w = neur.get_weights()
                weights.append(w[num])
            new_deltas.append(sum([w * d for w, d in zip(weights, deltas)]))
        return new_deltas

class NeuralNet:

    error_trhd = 0.010
    eps = 1e-4

    def __init__(self, num_layers, num_classes):
        self.layers = []
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.total_error = 0

    def train(self, inputs):
        
        output = []
        count = 10000
        self.init_layers(inputs[0])
        last_error = 0
        while(count > 0 and not self.isConverged(last_error)):
            last_error = self.total_error
            self.total_error = 0
            for input_pair in inputs:
                output = input_pair
                for layer in self.layers:
                    output = layer.train(output)
                    self.total_error += layer.getLayerError()
                
                #pdb.set_trace()
                last_delta = self.layers[-1].adjust_weights_def()
            
                
            #print(last_delta)
            #if self.num_layers > 1:
                for layer in reversed(self.layers[:-1]):
                    last_delta = layer.adjust_weights(last_delta)
            #print(self.total_error)
                    #print(last_delta)
            count -= 1
            
        print(count)
        #print(inputs)
        return output

    def init_layers(self, input_pair):
        import random
        
        output = input_pair
        for num in range(self.num_layers - 1):
            self.layers.append(self.createLayer(1, len(output[0])))
            output = self.layers[-1].train(output)
        self.layers.append(self.createLayer(self.num_classes - 1, len(output[0])))
                
    def isConverged(self, last_error):
        if last_error == 0:
            return False;
        return abs(abs(last_error - self.total_error) / last_error - NeuralNet.error_trhd) <= NeuralNet.eps

    def createLayer(self, num_neurs, num_inputs):
        return Layer(num_neurs, num_inputs)

    def get_weights(self):
        return [layer.get_weights() for layer in self.layers]
