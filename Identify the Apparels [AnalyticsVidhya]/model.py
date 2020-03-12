import numpy as np
import tensorflow as tf
import math

def mini_batches_X(X, mini_batch_size = 64):
    m = X.shape[0]                  # number of training examples
    mini_batches = []

    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batches.append(mini_batch_X)
    
    if m % mini_batch_size != 0:
        mini_batch_X = X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batches.append(mini_batch_X)
    
    return mini_batches



def random_mini_batches(X, Y, mini_batch_size = 64):
    m = X.shape[0]
    mini_batches = []
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def create_placeholders(n_H, n_W, n_C, n_Y):
    X = tf.placeholder(shape = (None, n_H, n_W, n_C), dtype = tf.float32, name = "X")
    Y = tf.placeholder(shape = (None, n_Y), dtype = tf.float32, name = "Y")
    return X, Y

def initializer_parameters(): #W1 shape = (4, 4, 1, 12) if using kaggle dataset
    W1 = tf.get_variable(shape = (4, 4, 4, 12), dtype = tf.float32, name = "W1", initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable(shape = (4, 4, 12, 48), dtype = tf.float32, name = "W2", initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W3 = tf.get_variable(shape = (3, 3, 48, 64), dtype = tf.float32, name = "W3", initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W4 = tf.get_variable(shape = (3, 3, 64, 64), dtype = tf.float32, name = "W4", initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    
    parameters = {"W1":W1,
                  "W2":W2,
                  "W3":W3,
                  "W4":W4,
                  }
   
    return parameters

def forward_propagation(X, parameters, reuse = None):
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']
    
    Z1 = tf.nn.conv2d(X, W1, strides = (1, 2, 2, 1), padding = "VALID", name = "Z1")
    A1 = tf.nn.relu(Z1, name = "A1")
    Z2 = tf.nn.conv2d(A1, W2, strides = (1, 1, 1, 1,), padding = "VALID", name = "Z2")
    A2 = tf.nn.relu(Z2, name = "A2")
    Z3 = tf.nn.max_pool(A2, ksize = (1, 2, 2, 1), strides = (1, 2, 2, 1), padding = "VALID", name = "Z3")
    A3 = tf.nn.relu(Z3, name = "A3")
    Z4 = tf.nn.conv2d(A3, W3, strides = (1, 1, 1, 1), padding = "VALID", name = "Z4")
    A4 = tf.nn.relu(Z4, name = "A4")
    Z5 = tf.nn.conv2d(A4, W4, strides = (1, 1, 1, 1), padding = "VALID", name = "Z5")
    A5 = tf.nn.relu(Z5, name = "A5")
    A5F = tf.contrib.layers.flatten(A5)
    F1 = tf.contrib.layers.fully_connected(A5F, num_outputs = 64, activation_fn = tf.nn.relu, reuse = reuse, scope = "F1")
    F2 = tf.contrib.layers.fully_connected(F1, num_outputs = 32, activation_fn = tf.nn.relu, reuse = reuse, scope = "F2")
    F3 = tf.contrib.layers.fully_connected(F2, num_outputs = 16, activation_fn = tf.nn.relu, reuse = reuse, scope = "F3")
    F4 = tf.contrib.layers.fully_connected(F3, num_outputs = 10, activation_fn = None, reuse = reuse, scope = "F4")
    
    return F4

def compute_cost(Z, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, logits = Z))
    
    return cost