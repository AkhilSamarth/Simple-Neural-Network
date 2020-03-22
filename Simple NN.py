'''
A simple neural network with a variable number of layers and nodes.
Handwritten Digits

By: Akhil Samarth
'''

from math import *
from random import *

# sigmoid
def sig(n):
    if n > 75:
        return 1
    elif n < -75:
        return 0
    return 1 / (1 + exp(-n))

# derivative of sigmoid
def dSig(n):
    if n > 75 or n < -75:
        return 0
    return exp(-n) / ((1 + exp(-n)) ** 2)

# neural network class
class Network:
    # container for activation, weights and biases
    class Node:
        def __init__(self):
            self.act = 0
            self.weights = []
            self.bias = 0

        def __str__(self):
            return 'Node: [act: {0}, bias: {1}, weights: {2}]'.format(self.act, self.bias, self.weights)

        __repr__ = __str__


    # object to hold the gradient of the cost function
    class Gradient:

        def __init__(self, del_bias, del_weight):
            self.del_bias = del_bias
            self.del_weight = del_weight

        # performs the given func component-wise on the two grads and returns a new grad
        def operate(grad1, grad2, func):
            # add biases
            del_bias1 = grad1.del_bias
            del_bias2 = grad2.del_bias
            del_bias_final = [[0 for j in range(len(del_bias1[i]))] for i in range(len(del_bias1))]

            # add weights
            del_weight1 = grad1.del_weight
            del_weight2 = grad2.del_weight
            del_weight_final = [[[0 for k in range(len(del_weight1[i][j]))] for j in range(len(del_weight1[i]))] for i in range(len(del_weight1))]

            for i in range(len(del_bias1)):
                for j  in range(len(del_bias1[i])):
                    del_bias_final[i][j] = func(del_bias1[i][j], del_bias2[i][j])
                    for k in range(len(del_weight1[i][j])):
                        del_weight_final[i][j][k] = func(del_weight1[i][j][k], del_weight2[i][j][k])

            return Network.Gradient(del_bias_final, del_weight_final)


        # performs the given func on each component and the given scalar
        def operate_scalar(grad, scalar, func):
            # add biases
            del_bias_orig = grad.del_bias
            del_bias_final = [[0 for j in range(len(del_bias_orig[i]))] for i in range(len(del_bias_orig))]

            # add weights
            del_weight_orig = grad.del_weight
            del_weight_final = [[[0 for k in range(len(del_weight_orig[i][j]))] for j in range(len(del_weight_orig[i]))] for i in range(len(del_weight_orig))]

            for i in range(len(del_bias_orig)):
                for j  in range(len(del_bias_orig[i])):
                    del_bias_final[i][j] = func(del_bias_orig[i][j], scalar)
                    for k in range(len(del_weight_orig[i][j])):
                        del_weight_final[i][j][k] = func(del_weight_orig[i][j][k], scalar)

            return Network.Gradient(del_bias_final, del_weight_final)


        # operator overloads
        
        def __add__(self, other):
            if isinstance(other, Network.Gradient):
                return Network.Gradient.operate(self, other, lambda x, y: x + y)


        def __sub__(self, other):
            if isinstance(other, Network.Gradient):
                return Network.Gradient.operate(self, other, lambda x, y: x - y)

        def __mul__(self, other):
            if isinstance(other, float) or isinstance(other, int):
                return Network.Gradient.operate_scalar(self, other, lambda x, y: x * y)

            
        def __truediv__(self, other):
            if isinstance(other, float) or isinstance(other, int):
                return Network.Gradient.operate_scalar(self, other, lambda x, y: x / y)


        def __str__(self):
            return 'Gradient:\nBiases: {0}\nWeights: {1}\n'.format(self.del_bias, self.del_weight)


    # layer_count is a list where len(layer_count) = num of layers, and each value is nodes in that layer
    def __init__(self, layer_counts):
        # each sublist contains the nodes for that layer
        self.layers = [[] for i in range(len(layer_counts))]

        # add nodes
        for i in range(len(layer_counts)):
            for j in range(layer_counts[i]):
                node = Network.Node()
                if i > 0:
                    node.weights = [0 for i in range(layer_counts[i - 1])]
                self.layers[i].append(node)


    # updates all node activations
    def update(self):
        # start from layer 1 and go to end
        for i in range(1, len(self.layers)):
            # update each node in this layer
            for node in self.layers[i]:
                # sum of weights * activations + bias
                weighted_sum = node.bias
                
                for j in range(len(self.layers[i - 1])):
                    weighted_sum += self.layers[i - 1][j].act * node.weights[j]

                node.act = sig(weighted_sum)


    # call at start to fill nodes with random values (range a to b, inclusive) and update
    def randomize(self, a = -1, b = 1):
        # loop through each node (except input layer) and assign random weight and bias
        for i in range(1, len(self.layers)):
            for node in self.layers[i]:
                node.bias = uniform(a, b)
                for i in range(len(node.weights)):
                    node.weights[i] = uniform(a, b)

        self.update()


    # takes in activation values for the input layer
    def input(self, in_layer):
        for i in range(len(self.layers[0])):
            self.layers[0][i].act = in_layer[i]

        self.update()


    # return the output layer activations as a list
    def output(self, in_layer=None):
        if in_layer != None:
            self.input(in_layer)
        return [node.act for node in self.layers[len(self.layers) - 1]]


    # returns the gradient of the cost function as a Gradient object
    def get_gradient(self, desired):
        layer_count = len(self.layers)
        
        # make sure desired is the same length as output
        if len(desired) != len(self.layers[layer_count - 1]):
            raise Exception('Given \'desired\' list is not the same length as output layer.')
        
        # partial derivatives of cost with respect to activations
        del_act = [[0 for j in range(len(self.layers[i]))] for i in range(layer_count)]

        # output layer activation derivatives
        out_layer = self.layers[layer_count - 1]
        for i in range(len(out_layer)):
            del_act[layer_count - 1][i] = 2 * (out_layer[i].act - desired[i])

        # all other activations (except input layer)
        for i in range(layer_count - 2, 0, -1):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]

            # loop through each node of next layer and add its contribution to del_act
            for j in range(len(next_layer)):
                # calculate weighted sum associated with this node
                weighted_sum = next_layer[j].bias
                for k in range(len(layer)):
                    weighted_sum += layer[k].act * next_layer[j].weights[k]

                # calculate derivative of sigmoid of this weighted sum
                del_sig = dSig(weighted_sum)

                # add this nodes contribution to del_act
                for k in range(len(layer)):
                    del_act[i][k] += del_act[i + 1][j] * del_sig * next_layer[j].weights[k]

        # partial derivatives of cost with respect to weights and biases
        del_bias = [[0 for j in range(len(self.layers[i]))] for i in range(layer_count)]
        del_weight = [[[0 for n in range(len(self.layers[i][j].weights))] for j in range(len(self.layers[i]))] for i in range(layer_count)]

        # go through all nodes (except input) and calculate weights and biases
        for i in range(1, layer_count):
            layer = self.layers[i]
            prev_layer = self.layers[i - 1]
            
            for j in range(len(layer)):
                # calculate weighted sum associated with this node
                weighted_sum = layer[j].bias
                for k in range(len(prev_layer)):
                    weighted_sum += prev_layer[k].act * layer[j].weights[k]

                # calculate derivative of sigmoid of this weighted sum
                del_sig = dSig(weighted_sum)

                # calculate bias
                del_bias[i][j] = del_act[i][j] * del_sig

                # calculate weights
                for k in range(len(layer[j].weights)):
                    del_weight[i][j][k] = del_act[i][j] * prev_layer[k].act

        return Network.Gradient(del_bias, del_weight)
    
        
    # backpropogate using the given list of correct output activations
    def backprop(self, desired):
        grad = self.get_gradient(desired)
        del_bias = grad.del_bias
        del_weight = grad.del_weight
        
        # update all weights and biases using calculated gradient
        for i in range(1, len(self.layers)):
            layer = self.layers[i]

            for j in range(len(layer)):
                layer[j].bias -= del_bias[i][j]

                for k in range(len(layer[j].weights)):
                    layer[j].weights[k] -= del_weight[i][j][k]

        # update network
        self.update()


    # backpropogate using the given gradient
    def backprop_grad(self, grad):
        del_bias = grad.del_bias
        del_weight = grad.del_weight
        
        # update all weights and biases using calculated gradient
        for i in range(1, len(self.layers)):
            layer = self.layers[i]

            for j in range(len(layer)):
                layer[j].bias -= del_bias[i][j]

                for k in range(len(layer[j].weights)):
                    layer[j].weights[k] -= del_weight[i][j][k]

        # update network
        self.update()

    # saves the network to a file
    def save(self, fileName, warning=True):
        if warning:
            # warn user before overwriting
            response = input('Warning: writing to file {0}.nn, file will be overwritten if it exists. Enter y to proceed: '.format(fileName))
            if response != 'y':
                print('Saving cancelled')
                return

        # gather data for writing
        layer_list = [len(layer) for layer in self.layers]
        in_layer = [node.act for node in self.layers[0]]

        file = open(fileName + '.nn', 'w')

        # write layer and input data
        file.write('')
        for num in layer_list:
            file.write('{0},'.format(num))
        file.write('\n')
        for num in in_layer:
            file.write('{0},'.format(num))
        file.write('\n')

        # format for weights and biases (1 node = 1 line): bias, weight0, weight1...
        # in order starting from layer 1
        for i in range(1, len(self.layers)):
            for j in range(len(self.layers[i])):
                node = self.layers[i][j]
                file.write('{0},'.format(node.bias))
                for weight in node.weights:
                    file.write('{0},'.format(weight))
                file.write('\n')
                
        file.close()
        print('Network saved to file {0}.nn'.format(fileName))


    # loads a new network from a file and returns it
    def load(fileName):
        file = open(fileName + '.nn', 'r')
        lines = file.readlines()

        # use first and second lines to create network
        layers = [int(x) for x in lines[0].split(',') if x != '\n']
        inputs = [float(x) for x in lines[1].split(',') if x != '\n']
        net = Network(layers)
        net.input(inputs)

        # loop through lines and fill nodes
        lines = lines[2:]
        layerIndex = 1
        nodeIndex = 0
        for line in lines:
            node = net.layers[layerIndex][nodeIndex]
            vals = [float(x) for x in line.split(',') if x != '\n']

            node.bias = vals[0]
            vals = vals[1:]
            node.weights = vals

            nodeIndex += 1
            if nodeIndex >= len(net.layers[layerIndex]):
                nodeIndex = 0
                layerIndex += 1

        net.update()
        return net
        
        
    def __str__(self):
        netStr = 'Network:\n'

        for i in range(len(self.layers)):
            netStr += 'Layer {0}: {1}\n'.format(i, self.layers[i])

        return netStr

    __repr__ = __str__
