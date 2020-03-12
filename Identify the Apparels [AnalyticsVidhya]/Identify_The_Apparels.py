import cv2
import os
import math
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
Y_train = np.load('Y_train.npy')

train = pd.read_csv('train/train.csv')
test = pd.read_csv('test/test.csv')

X_train = train.iloc[:, 0].values
Y_train = train.iloc[:, 1].values
X_test = test.iloc[:, 0].values

del test, train

Y_train = Y_train[:, np.newaxis]

X_train_ = []
X_test_ = []

count = 0
for i in X_train:#i = X_train[0]
    arr = Image.open()
    arr = cv2.imread(os.path.join('train', 'train', '%d.png' %(i)), -1)
    X_train_.append(arr)
    if not count % 100: print(count)
    count+=1
X_train = np.asarray(X_train_, dtype = np.float32)

for i in X_test:
    arr = ndimage.imread('test\\test\\%d.png' %(i))
    X_test_.append(arr)
    if not count % 100: print(count)
    count+=1
X_test = np.asarray(X_test_, dtype = np.float32)


from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder(categorical_features=[0])
Y_train = onehot.fit_transform(Y_train).toarray()

np.save("X_train.npy", X_train)
np.save("Y_train.npy", Y_train)
np.save("X_test.npy", X_test)

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

m, n_H, n_W, n_C = X_train.shape
n_Y = Y_train.shape[1]
costs = []

tf.reset_default_graph()

X, Y = create_placeholders(n_H, n_W, n_C, n_Y)
parameters = initializer_parameters()
Z = forward_propagation(X, parameters, reuse = None)
cost = compute_cost(Z, Y)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.000025).minimize(cost)

session = tf.Session()
session.run(tf.global_variables_initializer())
for i in range(100):
    minibatch_cost = 0.
    num_batches = int(m / 2048)
    minibatches = random_mini_batches(X_train, Y_train, mini_batch_size= 2048)
    for minibatch in minibatches:
        (minibatch_X, minibatch_Y) = minibatch
        _, temp_cost = session.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
        minibatch_cost += temp_cost
    print("Cost after %d epochs = %f" %(i+1, minibatch_cost))
    if i%5 == 0:
        costs.append(minibatch_cost)
        
parameters = session.run(parameters)

del X_test_, X_train_, arr, costs, count, i, m, minibatch, minibatch_X, minibatch_Y, minibatch_cost, minibatches, num_batches, temp_cost

#Y_train = onehot.inverse_transform(Y_train)

Y_test_pred = session.run(forward_propagation(X_test, parameters, reuse = True))
Y_test_pred = session.run(tf.nn.softmax(Y_test_pred))
Y_test_pred = session.run(tf.contrib.seq2seq.hardmax(Y_test_pred))
Y_test_pred = np.array([np.where(r == 1)[0][0] for r in Y_test_pred])
Y_test_pred = Y_test_pred.reshape((Y_test_pred.shape[0], 1))

output = {'id': np.arange(60001, 70001, 1),
          'label': Y_test_pred.reshape(Y_test_pred.shape[0])
          }
df = pd.DataFrame(output, columns = ['id', 'label'])
df.to_csv(r'Output.csv', index = None, header = True)
