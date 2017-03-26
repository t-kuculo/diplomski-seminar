from fann2 import libfann
import numpy as np





learning_rate = 0.7
desired_error = 0.0001
max_iterations = 100000
iterations_between_reports = 1000

#################### ANN that deals with LOCAL context ######################

# dummy X: 3 embeddings with 3 features
X = np.matrix('0.123 0.234 0.345;'+
              '0.234 0.345 0.456;'+
              '0.345 0.456 0.567)')

x_input = X.sum(axis=0)

n = len(x_input)
h = 10

local_ann = libfann.neural_net()
local_ann.create_standard(3, n, h, 1)
local_ann.set_learning_rate(learning_rate)
ann.set_activation_function_output(libfann.FANN_SIGMOID_SYMMETRIC)
