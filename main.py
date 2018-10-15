#import libraries and stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from NeuralNetwork import Layer, NeuralNetwork
from PSO import ParticleStructure, Swarm

#import mnist from local directory
data = pd.read_csv('E:\\Machine Learning\\NN\\iris.csv')

#data dimension measure
number_of_samples = data.shape[0]
number_of_features = data.shape[1]- 1 

#data_target_split
Y = np.array(data['species'])
Y = np.reshape(Y, (number_of_samples, 1))
X = np.array(data[data.columns[0:4]])

X = X.T

print(Y.shape)
#label vector regenerate
Y_Original = np.zeros((number_of_samples,3))
# print(Y_Original)
true_output_matrix = np.identity(3)
# print(true_output_matrix)
for i in range(number_of_samples):
    Y_Original[i] = Y_Original[i] + true_output_matrix[Y[i]]
Y = Y_Original.T


#initialize network
PSO_Net = NeuralNetwork([1, 8, 3])
PSO_Net.structure_define()
# print(PSO_Net)

num_epoch = 200
epoch = np.arange(1, num_epoch+1)
loss = np.zeros((num_epoch, 1))

tick = time.time()

for i in range(num_epoch):
    PSO_Net.forward_propagation(X)
#     print(PSO_Net.A_3.shape)
    loss[i] = PSO_Net.calculate_network_loss(Y)
    
    print(loss[i])
#     print(Y.shape)
    PSO_Net.backward_propagation(X, Y)
tock = time.time()

print("Runtime of Backprop: ", tock-tick)

plt.figure(1)
plt.semilogy(epoch, loss)
plt.xlabel('Epoch')
plt.ylabel('Error Amplitude in Log Dimension')
plt.show()
