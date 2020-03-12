#Importing necessary libraries
import math
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#Importing datasets using pandas
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Preparing Y_train and applying LabelEncoder()
Y_train = train['breed'].values
breedEncoder = LabelEncoder()
Y_train = breedEncoder.fit_transform(Y_train).reshape((6206, 1))
breedOneHot = OneHotEncoder(categorical_features = [0])
Y_train = breedOneHot.fit_transform(Y_train).toarray()

#X_$
X_train = train['id'].values
X_test = test['id'].values

del train, test

#Checking images for minimum image size
l, b = 400, 400
for i in X_train:
    x, y, z = cv2.imread('images_train\\'+i+'.jpg').shape
    if(l>x):l=x
    if(b>y):b=y
del b, i, l, x, y, z

#Creating Final X_train dataset
X_train_images = []
for i in X_train:
    X_train_images.append(cv2.resize(cv2.imread('images_train\\'+i+'.jpg'), (100, 100)))    
X_train = np.array(X_train_images)
del i, X_train_images

#Creating Final X_test dataset
X_test_images = []
for i in X_test:
    X_test_images.append(cv2.resize(cv2.imread('images_test\\'+i+'.jpg'), (100, 100)))
X_test = np.array(X_test_images)
del i, X_test_images

#Saving Final Numpy arrays
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("Y_train.npy", Y_train)

#Loading Final Numpy arrays
X_train = np.load("X_train.npy")
X_test =  np.load("X_test.npy")
Y_train = np.load("Y_train.npy")

X_test = np.array(X_test, dtype = np.float32)
X_train = np.array(X_train, dtype = np.float32)


def random_mini_batches_X(X, mini_batch_size = 64):
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batches.append(mini_batch_X)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batches.append(mini_batch_X)
    
    return mini_batches


def random_mini_batches(X, Y, mini_batch_size = 64):
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


############################TENSORFLOW MODEL###################################

#Create placeholders
def create_placeholders(n_H, n_W, n_C, n_Y):

    X = tf.placeholder(dtype = tf.float32, shape = (None, n_H, n_W, n_C), name = "X")
    Y = tf.placeholder(dtype = tf.float32, shape = (None, n_Y), name = "Y")

    return X, Y

def initialize_parameters(n_H, n_W, n_C, n_Y):
    W1 = tf.get_variable(name = "W1", dtype = tf.float32, shape = (4, 4, 3, 24), initializer = tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable(name = "W2", dtype = tf.float32, shape = (4, 4, 24, 96), initializer = tf.contrib.layers.xavier_initializer())
    W3 = tf.get_variable(name = "W3", dtype = tf.float32, shape = (6, 6, 96, 120), initializer = tf.contrib.layers.xavier_initializer())
    W4 = tf.get_variable(name = "W4", dtype = tf.float32, shape = (4, 4, 120, 120), initializer = tf.contrib.layers.xavier_initializer())
    W5 = tf.get_variable(name = "W5", dtype = tf.float32, shape = (3, 3, 120, 120), initializer = tf.contrib.layers.xavier_initializer())
    W6 = tf.get_variable(name = "W6", dtype = tf.float32, shape = (4, 4, 120, 150), initializer = tf.contrib.layers.xavier_initializer())
    W7 = tf.get_variable(name = "W7", dtype = tf.float32, shape = (1, 1, 150, 640), initializer = tf.contrib.layers.xavier_initializer())
    
    parameters = {"W1":W1,
                  "W2":W2,
                  "W3":W3,
                  "W4":W4,
                  "W5":W5,
                  "W6":W6,
                  "W7":W7
                  }

    return parameters


def forward_propagation(X, parameters, n_H, n_W, n_C, n_Y, reuse = None):

    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']
    W5 = parameters['W5']
    W6 = parameters['W6']
    W7 = parameters['W7']
    
    Z1 = tf.nn.conv2d(X, W1, strides = (1, 2, 2, 1), padding = "VALID", name = "Z1")
    A1 = tf.nn.relu(Z1, name = "A1")
    Z2 = tf.nn.conv2d(A1, W2, strides = (1, 2, 2, 1), padding = "VALID", name = "Z2")
    A2 = tf.nn.relu(Z2, name = "A2")
    Z3 = tf.nn.conv2d(A2, W3, strides = (1, 1, 1, 1), padding = "VALID", name = "Z3")
    A3 = tf.nn.relu(Z3, name = "A3")
    Z4 = tf.nn.conv2d(A3, W4, strides = (1, 2, 2, 1), padding = "VALID", name = "Z4")
    A4 = tf.nn.relu(Z4, name = "A4")
    Z5 = tf.nn.conv2d(A4, W5, strides = (1, 1, 1, 1), padding = "VALID", name = "Z5")
    A5 = tf.nn.relu(Z5, name = "A5")
    Z6 = tf.nn.conv2d(A5, W6, strides = (1, 1, 1, 1), padding = "VALID", name = "Z6")
    A6 = tf.nn.relu(Z6, name = "A6")
    Z7 = tf.nn.conv2d(A6, W7, strides = (1, 1, 1, 1), padding = "VALID", name = "Z7")
    A7 = tf.nn.relu(Z7, name = "A7")
    A7F = tf.contrib.layers.flatten(A7)
    F1 = tf.contrib.layers.fully_connected(A7F, 480, activation_fn = tf.nn.relu, reuse = reuse, scope = "FC1")
    F2 = tf.contrib.layers.fully_connected(F1, 120, activation_fn = tf.nn.relu, reuse = reuse, scope = "FC2")
    F3 = tf.contrib.layers.fully_connected(F2, 60, activation_fn = tf.nn.relu, reuse = reuse, scope = "FC3")
    Z = tf.contrib.layers.fully_connected(F3, 10, activation_fn = None, reuse = reuse, scope = "FC4")

    return Z
    
def compute_cost(Z, Y):

    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, logits = Z))

    return cost

m, n_H, n_W, n_C = X_train.shape
n_Y = Y_train.shape[1]
costs = []

tf.reset_default_graph()

from sklearn.model_selection import train_test_split
X_train, X_cv, Y_train, Y_cv = train_test_split(X_train, Y_train, test_size = 0.113)

X, Y = create_placeholders(n_H, n_W, n_C, n_Y)
parameters = initialize_parameters(n_H, n_W, n_C, n_Y)
Z = forward_propagation(X, parameters, n_H, n_W, n_C, n_Y)
cost = compute_cost(Z, Y)
optimizer = tf.train.RMSPropOptimizer(learning_rate = 0.000025).minimize(cost)

session = tf.Session()
session.run(tf.global_variables_initializer())

for i in range(30):
    minibatch_cost = 0.
    num_batches = int(m / 64)
    minibatches = random_mini_batches(X_train, Y_train, mini_batch_size= 64)
    
    for minibatch in minibatches:
        (minibatch_X, minibatch_Y) = minibatch
        _, temp_cost = session.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
        minibatch_cost += temp_cost
    
    print("Cost after %d epochs = %f" %(i+1, minibatch_cost))
    if i%5 == 0:
        costs.append(minibatch_cost)




parameters = session.run(parameters)

    
Y_pred_cv = session.run(forward_propagation(X_cv, parameters, n_H, n_W, n_C, n_Y, reuse = True))
Y_pred_cv = session.run(tf.nn.softmax(Y_pred_cv))
Y_pred_cv = session.run(tf.contrib.seq2seq.hardmax(Y_pred_cv))

Y_pred_cv = np.array([np.where(r == 1)[0][0] for r in Y_pred_cv])
Y_pred_cv = Y_pred_cv.reshape((Y_pred_cv.shape[0], 1))

Y_cv = breedOneHot.inverse_transform(Y_cv)

cm = 0
for i in range(len(Y_cv)):
    if Y_cv[i][0] == Y_pred_cv[i][0]: cm+=1

Y_class = []
for i in Y_pred_test_:
    if i in range(11, 16): Y_class.append(1)
    else: Y_class.append(2)

Y_class = np.array(Y_class)

#tf.saved_model.simple_save(session, "tfmodel", inputs = None, outputs=None)

output = {'id': test.iloc[:, 0].values,
          'class_name': Y_pred_test_,
          'breed': Y_class
          }
df = pd.DataFrame(output, columns = ['id', 'class_name', 'breed'])
df.to_csv(r'Output.csv', index = None, header = True)    