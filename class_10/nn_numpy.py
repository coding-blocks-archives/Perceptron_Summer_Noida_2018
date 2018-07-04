# Teaching a 3 level neural network to work as Full Adder

#import numpy for maths, pandas for reading data from text
from numpy import random, exp, array, dot
import pandas as pd
from matplotlib import pyplot as plt

# Neural Network class definition
class NeuralNetwork():
  def __init__(self, gateInput, gateOutput, ):

    # seed the random generator for easier debugging
    random.seed(1)

    # Save all variables in self for future references
    self.gateInput = gateInput
    self.gateOutput = gateOutput
    self.input_shape = (1,3)
    self.output_shape = (1,2)
    self.layer_1_nodes = 5
    self.layer_2_nodes = 5
    self.layer_3_nodes = 5

    # Generate weights with value between -1 to 1 so that mean is overall 0
    self.weights_1 = 2 * random.random((self.input_shape[1], self.layer_1_nodes)) - 1
    self.weights_2 = 2 * random.random((self.layer_1_nodes, self.layer_2_nodes)) - 1
    self.weights_3 = 2 * random.random((self.layer_2_nodes, self.layer_3_nodes)) - 1
    self.out_weights = 2 * random.random((self.layer_3_nodes, self.output_shape[1])) - 1

  # Sigmoid function gives a value between 0 and 1
  def sigmoid(self, x):
    return 1 / (1 + exp(-x))

  # Reversed Sigmoid by derivating the value
  def sigmoid_derivative(self, x):
    return x * (1 - x)

  def think(self, x):
    # Multiply the input with weights and find its sigmoid activation for all layers
    layer1 = self.sigmoid(dot(x, self.weights_1))
    layer2 = self.sigmoid(dot(layer1, self.weights_2))
    layer3 = self.sigmoid(dot(layer2, self.weights_3))
    output = self.sigmoid(dot(layer3, self.out_weights))
    return output

  def train(self, num_steps):
    errors = []
    for x in range(num_steps):
      # Same as code of thinking
      layer1 = self.sigmoid(dot(self.gateInput, self.weights_1))
      layer2 = self.sigmoid(dot(layer1, self.weights_2))
      layer3 = self.sigmoid(dot(layer2, self.weights_3))
      output = self.sigmoid(dot(layer3, self.out_weights))

      # What is the error?
      outputError = self.gateOutput - output
      error = (outputError**2).sum() / (2 * gateOutput.shape[0])
      errors.append(error)

      # Find delta, i.e. Product of Error and derivative of next layer
      delta = outputError * self.sigmoid_derivative(output)
      
      # Multiply with transpose of last layer
      # to invert the multiplication we did to get layer 
      out_weights_adjustment = dot(layer3.T, delta)
      
      # Apply the out_weights_adjustment
      self.out_weights += out_weights_adjustment
      
      # Procedure stays same, but the error now is the product of current weight and
      # Delta in next layer
      delta = dot(delta, self.out_weights.T) * self.sigmoid_derivative(layer3)
      weight_3_adjustment = dot(layer2.T, delta)
      self.weights_3 += weight_3_adjustment

      delta = dot(delta, self.weights_3.T) * self.sigmoid_derivative(layer2)
      weight_2_adjustment = dot(layer1.T, delta)
      self.weights_2 += weight_2_adjustment

      delta = dot(delta, self.weights_2.T) * self.sigmoid_derivative(layer1)
      weight_1_adjustment = dot(self.gateInput.T, delta)
      self.weights_1 += weight_1_adjustment
    return errors

if __name__ == '__main__':
  file = pd.read_csv("../../data/dataset.txt", delimiter=',')

  dataset = file.values

  gateInput = dataset[:,:3]
  gateOutput = dataset[:,3:]

  neural_network = NeuralNetwork(gateInput, gateOutput)

  err = neural_network.train(6000)
  plt.plot(err)
  plt.show()
  # Should be 0 , 1
  print(neural_network.think([[0,1,1]]))