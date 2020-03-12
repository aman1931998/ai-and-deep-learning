#Importing necessary libraries
import math
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

#Importing datasets using pandas
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Preparing Y_train and applying OneHotEncoder()
Y_train = train['breed'].replace(to_replace = {11:1, 12:2, 13:3, 14:4, 15:5, 21:6, 22:7, 23:8, 24:9, 25:10})
Y_train = Y_train.values.reshape((Y_train.shape[0], 1))
onehot = OneHotEncoder()
Y_train = np.array(onehot.fit_transform(Y_train).toarray(), dtype = np.float32)

#X_$
X_train = train['id'].values
X_test = test['id'].values

del train, test

#Checking images for necessary image_size
l, b = 0, 0
for i in X_train:
    x, y, z = cv2.imread('images_train\\'+i+'.jpg').shape
    l+=x
    b+=y
l/=6206
b/=6206
del b, i, l, x, y, z

#l, b = 250, 200

#Creating Final X_train dataset
X_train_images = []
for i in X_train:
    X_train_images.append(cv2.resize(cv2.imread('images_train\\'+i+'.jpg'), (224, 224)))    #(250, 200)
X_train = np.array(X_train_images, dtype = np.float32)
del i, X_train_images

#Creating Final X_test dataset
X_test_images = []
for i in X_test:
    X_test_images.append(cv2.resize(cv2.imread('images_test\\'+i+'.jpg'), (224, 224)))    #(250, 200)
X_test = np.array(X_test_images, dtype = np.float32)
del i, X_test_images

#Saving Final Numpy arrays
np.save("npy_files/X_train.npy", X_train)
np.save("npy_files/X_test.npy", X_test)
np.save("npy_files/Y_train.npy", Y_train)

#Loading Final Numpy arrays
X_train = np.load("npy_files/X_train.npy")
X_test =  np.load("npy_files/X_test.npy")
Y_train = np.load("npy_files/Y_train.npy")

#####################TEMP -> CV Check
X_train, X_cv, Y_train, Y_cv = train_test_split(X_train, Y_train, test_size = 0.113760)


def random_mini_batches_X(X, mini_batch_size = 64):
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
    W1 = tf.get_variable(name = "W1", dtype = tf.float32, shape = (11, 11, 3, 16), initializer = tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable(name = "W2", dtype = tf.float32, shape = (11, 11, 16, 48), initializer = tf.contrib.layers.xavier_initializer())
    W3 = tf.get_variable(name = "W3", dtype = tf.float32, shape = (6, 6, 48, 96), initializer = tf.contrib.layers.xavier_initializer())
    W4 = tf.get_variable(name = "W4", dtype = tf.float32, shape = (4, 4, 96, 256), initializer = tf.contrib.layers.xavier_initializer())
    W5 = tf.get_variable(name = "W5", dtype = tf.float32, shape = (4, 4, 256, 256), initializer = tf.contrib.layers.xavier_initializer())

    W6 = tf.get_variable(name = "W6", dtype = tf.float32, shape = (4, 4, 256, 256), initializer = tf.contrib.layers.xavier_initializer())

    W7 = tf.get_variable(name = "W7", dtype = tf.float32, shape = (7, 7, 256, 256), initializer = tf.contrib.layers.xavier_initializer())

    parameters = {"W1":W1,
                  "W2":W2,
                  "W3":W3,
                  "W4":W4,
                  "W5":W5,
                  "W6":W6,
                  "W7":W7,
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
    
    Z1 = tf.nn.conv2d(X, W1, strides = (1, 1, 1, 1), padding = "SAME", name = "Z1")
    A1 = tf.nn.relu(Z1, name = "A1")

    Z2 = tf.nn.conv2d(A1, W2, strides = (1, 1, 1, 1), padding = "SAME", name = "Z2")
    A2 = tf.nn.relu(Z2, name = "A2")
    
    Z3 = tf.nn.max_pool(A2, ksize = (1, 2, 2, 1), strides = (1, 2, 2, 1), padding = "VALID", name = "Z3")
    A3 = tf.nn.relu(Z3, name = "A3")


    Z4 = tf.nn.conv2d(A3, W3, strides = (1, 1, 1, 1), padding = "SAME", name = "Z4")
    A4 = tf.nn.relu(Z4, name = "A4")
    Z5 = tf.nn.max_pool(A4, ksize = (1, 2, 2, 1), strides = (1, 2, 2, 1), padding = "VALID", name = "Z5")
    A5 = tf.nn.relu(Z5, name = "A5")

    Z6 = tf.nn.conv2d(A5, W4, strides = (1, 1, 1, 1), padding = "SAME", name = "Z6")
    A6 = tf.nn.relu(Z6, name = "A6")

    Z7 = tf.nn.max_pool(A6, ksize = (1, 2, 2, 1), strides = (1, 2, 2, 1), padding = "VALID", name = "Z7")
    A7 = tf.nn.relu(Z7, name = "A7")

    Z8 = tf.nn.conv2d(A7, W5, strides = (1, 1, 1, 1), padding = "SAME", name = "Z8")
    A8 = tf.nn.relu(Z8, name = "A8")

    Z9 = tf.nn.max_pool(A8, ksize = (1, 2, 2, 1), strides = (1, 2, 2, 1), padding = "VALID", name = "Z9")
    A9 = tf.nn.relu(Z9, name = "A9")

    Z10 = tf.nn.conv2d(A9, W6, strides = (1, 1, 1, 1), padding = "SAME", name = "Z10")
    A10 = tf.nn.relu(Z10, name = "A10")

    Z11 = tf.nn.max_pool(A10, ksize = (1, 2, 2, 1), strides = (1, 2, 2, 1), padding = "VALID", name = "Z11")
    A11 = tf.nn.relu(Z11, name = "A11")

    Z12 = tf.nn.conv2d(A11, W7, strides = (1, 1, 1, 1), padding = "VALID", name = "Z12")
    A12 = tf.nn.relu(Z12, name = "A12")

    A12F = tf.contrib.layers.flatten(A12)

    F1 = tf.contrib.layers.fully_connected(A12F, 256, activation_fn = tf.nn.relu, reuse = reuse, scope = "FC1")
    F2 = tf.contrib.layers.fully_connected(F1, 96, activation_fn = tf.nn.relu, reuse = reuse, scope = "FC2")
    F3 = tf.contrib.layers.fully_connected(F2, 32, activation_fn = tf.nn.relu, reuse = reuse, scope = "FC3")
    Z = tf.contrib.layers.fully_connected(F3, 10, activation_fn = None, reuse = reuse, scope = "FC4")

    return Z
    
def compute_cost(Z, Y):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, logits = Z))

    return cost

#Clearing up RAM
del X_test

m, n_H, n_W, n_C = X_train.shape
n_Y = Y_train.shape[1]
costs = []

tf.reset_default_graph()

X, Y = create_placeholders(n_H, n_W, n_C, n_Y)
parameters = initialize_parameters(n_H, n_W, n_C, n_Y)
Z = forward_propagation(X, parameters, n_H, n_W, n_C, n_Y)
cost = compute_cost(Z, Y)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.000025).minimize(cost)

session = tf.Session()
session.run(tf.global_variables_initializer())

#writer = tf.summary.FileWriter(r"D:\Datsets\github\data-science-and-machine-learning\MachineHack_Who_Let_The_Dogs_Out_Pets_Breed_Classification_Hackathon\tmp\tflogs", session.graph)

for i in range(5):
    minibatch_cost = 0.
    num_batches = int(m / 32)
    minibatches = random_mini_batches(X_train, Y_train, mini_batch_size= 32)

    count_mini = 0
    for minibatch in minibatches:
        (minibatch_X, minibatch_Y) = minibatch
        _, temp_cost = session.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
        minibatch_cost += temp_cost
        if not count_mini % 5: print("Minibatch count %d" %count_mini)
        count_mini+=1
    
    print("Cost after %d epochs = %f" %(i+1, minibatch_cost))
    if i%5 == 0:
        costs.append(minibatch_cost)

#writer.close()

parameters = session.run(parameters)

del X_train, Y_train, costs, i, minibatch, minibatch_X, minibatch_Y, minibatch_cost, minibatches, temp_cost
del count_mini, num_batches

X_cv_minibatches = random_mini_batches_X(X_cv, mini_batch_size = 32)


l = []
for i in X_cv_minibatches:
    Y_pred_cv = session.run(forward_propagation(i, parameters, n_H, n_W, n_C, n_Y, reuse = True))
    Y_pred_cv = session.run(tf.nn.softmax(Y_pred_cv))
    Y_pred_cv = session.run(tf.contrib.seq2seq.hardmax(Y_pred_cv))    
    Y_pred_cv = onehot.inverse_transform(Y_pred_cv)
    l.append(Y_pred_cv)

Y_pred_cv = np.concatenate((l[0], l[1], l[2], l[3], l[4], 
                              l[5], l[6], l[7], l[8], l[9], 
                              l[10], l[11], l[12], l[13], l[14], 
                              l[15], l[16], l[17], l[18], l[19]), axis = 0)

Y_cv = onehot.inverse_transform(Y_cv)

cm = 0
for i in range(len(Y_cv)):
    if Y_cv[i][0] == Y_pred_cv[i][0]: cm+=1


Y_class = []
for i in Y_pred_test:
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