#import libraries and stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

#initialize the network

class Layer:
    def __init__(self, layer_parameter):
        if(layer_parameter[1]!='none'): 
            self.activations_for_this_layer = np.zeros(layer_parameter[0])
            self.number_of_nodes = layer_parameter[0]
            self.activation_function = layer_parameter[1]
            self.weights_from_previous_layer = layer_parameter[2]
            self.biases_for_this_layer = layer_parameter[3]
    
        
class NeuralNetwork:
    
    #constructor
    def __init__(self, network_parameter):
        self.number_of_hidden_layers = network_parameter[0]
        self.hidden_layer_size = network_parameter[1]
        self.output_size = network_parameter[2]
        self.layers = []
        self.final_output = 0
        
    #activation functions
    def ReLU(self, x):
        return np.maximum(x, 0)

    def DReLU(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x
    
    def sigmoid(self, x):
        return np.divide(1, np.add(1, np.exp(np.negative(x))))
    
    def Dsigmoid(self, x):
        return sigmoid(x) * (1-sigmoid(x))
    
    def softmax(self, x):
        exp = np.exp(x)
        if isinstance(x[0], np.ndarray):
            return exp/np.sum(exp, axis=1, keepdims=True)
        else:
            return exp/np.sum(exp, keepdims=True)
    
    def tanh(self, x):
        return np.divide(np.exp(x)-np.exp(np.negative(x)) , np.exp(x)+np.exp(np.negative(x)))
    
    def Dtanh(self, x):
        return (1-self.tanh(x)**2)
    
    #network structure define
    def structure_define(self):
        
        #create input layer
        input_layer = Layer([4, 'none', [], []])
        self.layers.append(input_layer)

        #create hidden layer 1
        #np.random.seed(1)
        hidden_layer_1_weights = np.random.randn(8, 4) * 0.01
        hidden_layer_1_biases = np.random.randn(8, 1)
        hidden_layer_1 = Layer([8, 'ReLU', hidden_layer_1_weights, hidden_layer_1_biases])
        self.layers.append(hidden_layer_1)

        # #create hidden layer 2
        # #np.random.seed(2)
        # hidden_layer_2_weights = np.random.randn(16, 16) * 0.01
        # hidden_layer_2_biases = np.random.randn(16, 1)
        # hidden_layer_2 = Layer([16, 'ReLU', hidden_layer_2_weights, hidden_layer_2_biases])
        # self.layers.append(hidden_layer_2)

        #create output layer
        #np.random.seed(3)
        output_layer_weights = np.random.randn(3, 8) * 0.01
        output_layer_biases = np.random.randn(3, 1)
        output_layer = Layer([3, 'Sigmoid', output_layer_weights, output_layer_biases])
        self.layers.append(output_layer)
        print("HLLLl")
            
    def forward_propagation(self, X): 
        
        # Hidden Layer 1
        self.W_1 = self.layers[1].weights_from_previous_layer
        self.B_1 = self.layers[1].biases_for_this_layer
        self.Z_1 = np.dot(self.W_1, X) + self.B_1
        self.A_1 = self.ReLU(self.Z_1)
        
        # #Hidden Layer 2
        # self.W_2 = self.layers[2].weights_from_previous_layer
        # self.B_2 = self.layers[2].biases_for_this_layer
        # self.Z_2 = np.dot(self.W_2, self.A_1) + self.B_2
        # self.A_2 = self.ReLU(self.Z_2)
        
        #Hidden Layer 3
        self.W_3 = self.layers[2].weights_from_previous_layer
        self.B_3 = self.layers[2].biases_for_this_layer
        self.Z_3 = np.dot(self.W_3, self.A_1) + self.B_3
        self.A_3 = self.sigmoid(self.Z_3)
    
    def calculate_network_loss(self, Y):
        # loss = np.sum((-1) * (np.dot(Y, np.log(self.A_3.T))) + np.dot((1-Y),(np.log(1-self.A_3.T))))
        # loss = abs(loss / 150)
        loss = 0
        for i in range(3):
            arr1 = self.A_3[i].T
            arr1[arr1 == 0] = 0.000000000001 #remove zeros to avoid log 0 which will produce nan
            loss_1 = np.dot(Y[i], np.log(arr1))
            arr2 = 1-self.A_3[i].T
            arr2[arr2 == 0] = 0.000000000001 #remove zeros to avoid log 0 which will produce nan
            loss_2 = np.dot((1-Y[i]),np.log(arr2))
            loss_3 = -loss_1 - loss_2
            loss_3 = abs(loss_3 / 150)
            loss = loss + loss_3
        return loss
        
    def backward_propagation(self, X, Y):
        
        #Output Layer
        self.dZ_3 = self.A_3 - Y
        self.dW_3 = (1/150) * np.dot(self.dZ_3, self.A_1.T)
        self.dB_3 = (1/150) * np.sum(self.dZ_3, axis=1, keepdims=True)
        
        # #Hidden Layer 2
        # self.dZ_2 = np.multiply(np.dot(self.W_3.T, self.dZ_3), self.DReLU(self.Z_2))
        # self.dW_2 = (1/150) * np.dot(self.dZ_2, self.A_1.T)
        # self.dB_2 = (1/150) * np.sum(self.dZ_2, axis=1, keepdims=True)
        
        #Hidden Layer 1
        self.dZ_1 = np.multiply(np.dot(self.W_3.T, self.dZ_3), self.DReLU(self.Z_1))
        self.dW_1 = (1/150) * np.dot(self.dZ_1, X.T)
        self.dB_1 = (1/150) * np.sum(self.dZ_1, axis=1, keepdims=True)
        
        #Weight Update
        self.layers[1].weights_from_previous_layer = self.layers[1].weights_from_previous_layer - 0.01 * self.dW_1
        self.layers[1].biases_for_this_layer = self.layers[1].biases_for_this_layer - 0.001 * self.dB_1
        
        # self.layers[2].weights_from_previous_layer = self.layers[2].weights_from_previous_layer - 0.01 * self.dW_2
        # self.layers[2].biases_for_this_layer = self.layers[2].biases_for_this_layer - 0.001 * self.dB_2
        
        self.layers[2].weights_from_previous_layer = self.layers[2].weights_from_previous_layer - 0.01 * self.dW_3
        self.layers[2].biases_for_this_layer = self.layers[2].biases_for_this_layer - 0.001 * self.dB_3