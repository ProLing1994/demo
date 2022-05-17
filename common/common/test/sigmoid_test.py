import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0/(1 + np.exp(- 12.0 * (x - 0.8) ))
 
sigmoid_inputs = np.arange(0, 1.6, 0.01)
sigmoid_outputs = sigmoid(sigmoid_inputs)
print("Sigmoid Function Input :: {}".format(sigmoid_inputs))
print("Sigmoid Function Output :: {}".format(sigmoid_outputs))
 
plt.plot(sigmoid_inputs,sigmoid_outputs)
plt.xlabel("Sigmoid Inputs")
plt.ylabel("Sigmoid Outputs")
plt.show()