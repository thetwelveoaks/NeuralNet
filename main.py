import utility
import entity

dataset = utility.readfile('spambase.data')
splited_data = utility.splitdata(dataset)
training = [utility.normalise(item) for item in splited_data[0]]
validation = [utility.normalise(item) for item in splited_data[1]]
testing = [utility.normalise(item) for item in splited_data[2]]

class Classifier(entity.Neuron):
    eps = 1e-5
    def calculate_err(self, output):
        if (self.input_pair[1] == 0 and output - 1 / 2 > Classifier.eps) \
            or (self.input_pair[1] == 1 and output - 1 / 2 <= Classifier.eps):
            return 1
        return 0

    def validate(self):
        return self.error == self.input_pair[1]

class ClassifierLayer(entity.Layer):
    
    def createNeuron(self, num_inputs):
        return Classifier(num_inputs, entity.Layer.alpha)

class ClassifierNet(entity.NeuralNet):

    def createLayer(self, num_neurs, num_inputs):
        return ClassifierLayer(num_neurs, num_inputs)
    
classifier_net = ClassifierNet(1, 2)
classifier_net.train(training)
print(classifier_net.get_weights())
