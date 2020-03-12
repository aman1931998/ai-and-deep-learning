#importing necessary modules
import math
import os
import tensorflow as tf
import numpy as np
import warnings
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

#Gathering the list of images
parasitized, uninfected = [], []
for _, _, k in os.walk("cell_images/Parasitized/"):
    parasitized+=k    
for _, _, k in os.walk("cell_images/Uninfected/"):
    uninfected+=k
uninfected.remove('Thumbs.db')
parasitized.remove('Thumbs.db')

del k

#Loading the image to determine the image size
count = 0
h1, h2, w1, w2 = 999, 999, 999, 999
for i in parasitized:
    x = cv2.imread('cell_images\\Parasitized\\'+i).shape
    if(h1>x[0]): h1=x[0]
    if(w1>x[1]): w1=x[1]    
    count+=1
    if(count%1000==0): print(count)
for i in uninfected:
    x = cv2.imread('cell_images\\Uninfected\\'+i).shape
    if(h2>x[0]): h2=x[0]
    if(w2>x[1]): w2=x[1]
    count+=1
    if(count%1000==0): print(count)

del count, h1, h2, i, w1, w2, x
#Result
#Average file size will be (h, w) = (64, 64)

#Creating image dataset with data augmentation
count = 0
uninfected_image_array, parasitized_image_array = [], []
for i in parasitized:
    img = Image.open('cell_images\\Parasitized\\'+i)
    img = img.resize((64, 64))
#    rotate_45 = np.array(img.rotate(45))
#    rotate_75 = np.array(img.rotate(75))
#    blur = cv2.blur(np.array(img), (10, 10))
    img = np.array(img)
    parasitized_image_array+=[img] #[blur, rotate_45, rotate_75] #
    count+=1
    if(count%1000==0): print(count)
for i in uninfected:
    img = Image.open('cell_images\\Uninfected\\'+i)
    img = img.resize((64, 64))
#    rotate_45 = np.array(img.rotate(45))
#    rotate_75 = np.array(img.rotate(75))
#    blur = cv2.blur(np.array(img), (10, 10))
    img = np.array(img)
    uninfected_image_array+=[img] #[blur, rotate_45, rotate_75] #
    count+=1
    if(count%1000==0): print(count)

uninfected_image_array = np.array(uninfected_image_array, dtype = np.float32)
parasitized_image_array = np.array(parasitized_image_array, dtype = np.float32)

del count, i, img, parasitized, uninfected #blur, rotate_45, rotate_75



#np.save("npy_files/parasitized.npy", parasitized_image_array)
#np.save("npy_files/uninfected.npy", uninfected_image_array)
#
#parasitized_image_array = np.load("npy_files/parasitized.npy")
#uninfected_image_array = np.load("npy_files/uninfected.npy")

X_dataset = np.concatenate((parasitized_image_array, uninfected_image_array), axis = 0)
Y_dataset = np.concatenate((np.ones((13779, 1), dtype = np.float32), np.zeros((13779, 1), dtype = np.float32)), axis = 0)

del parasitized_image_array, uninfected_image_array

np.save("npy_files/X_dataset.npy", X_dataset)
np.save("npy_files/Y_dataset.npy", Y_dataset)

X_dataset = np.load("npy_files/X_dataset.npy")
Y_dataset = np.load("npy_files/Y_dataset.npy")

X_train, X_test, Y_train, Y_test = train_test_split(X_dataset, Y_dataset, train_size = 0.870895, random_state = 44)

del X_dataset, Y_dataset

np.save("npy_files/X_train.npy", X_train)
np.save("npy_files/Y_train.npy", Y_train)
np.save("npy_files/X_test.npy", X_test)
np.save("npy_files/Y_test.npy", Y_test)

X_train = np.load("npy_files/X_train.npy")
Y_train = np.load("npy_files/Y_train.npy")
X_test = np.load("npy_files/X_test.npy")
Y_test = np.load("npy_files/Y_test.npy")

################################MODEL##########################################
def mini_batches_X(X, mini_batch_size = 64):
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitioning
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

def create_placeholders(n_H, n_W, n_C, n_Y):

    X = tf.placeholder(dtype = tf.float32, shape = (None, n_H, n_W, n_C), name = "X")
    Y = tf.placeholder(dtype = tf.float32, shape = (None, n_Y), name = "Y")

    return X, Y

def initialize_parameters(n_H, n_W, n_C, n_Y):
    W1 = tf.get_variable(name = "W1", dtype = tf.float32, shape = (4, 4, 3, 16), initializer = tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable(name = "W2", dtype = tf.float32, shape = (4, 4, 16, 48), initializer = tf.contrib.layers.xavier_initializer())
    W3 = tf.get_variable(name = "W3", dtype = tf.float32, shape = (4, 4, 48, 96), initializer = tf.contrib.layers.xavier_initializer())
    W4 = tf.get_variable(name = "W4", dtype = tf.float32, shape = (4, 4, 96, 192), initializer = tf.contrib.layers.xavier_initializer())
    W5 = tf.get_variable(name = "W5", dtype = tf.float32, shape = (2, 2, 192, 192), initializer = tf.contrib.layers.xavier_initializer())
    
    parameters = {"W1":W1,
                  "W2":W2,
                  "W3":W3,
                  "W4":W4,
                  "W5":W5
                  }

    return parameters


def forward_propagation(X, parameters, n_H, n_W, n_C, n_Y, reuse = None):

    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']
    W5 = parameters['W5']
    
    Z1 = tf.nn.conv2d(X, W1, strides = (1, 2, 2, 1), padding = "VALID", name = "Z1")
    A1 = tf.nn.relu(Z1, name = "A1")

    Z2 = tf.nn.conv2d(A1, W2, strides = (1, 1, 1, 1), padding = "VALID", name = "Z2")
    A2 = tf.nn.relu(Z2, name = "A2")

    Z3 = tf.nn.max_pool(A2, ksize = (1, 2, 2, 1), strides = (1, 2, 2, 1), padding = "VALID", name = "Z3")
    A3 = tf.nn.relu(Z3, name = "A3")

    Z4 = tf.nn.conv2d(A3, W3, strides = (1, 2, 2, 1), padding = "VALID", name = "Z4")
    A4 = tf.nn.relu(Z4, name = "A4")

    Z5 = tf.nn.conv2d(A4, W4, strides = (1, 2, 2, 1), padding = "VALID", name = "Z5")
    A5 = tf.nn.relu(Z5, name = "A5")

    Z6 = tf.nn.conv2d(A5, W5, strides = (1, 1, 1, 1), padding = "VALID", name = "Z6")
    A6 = tf.nn.relu(Z6, name = "A6")
    A6F = tf.contrib.layers.flatten(A6)
    F1 = tf.contrib.layers.fully_connected(A6F, 96, activation_fn = tf.nn.relu, reuse = reuse, scope = "FC1")
    F2 = tf.contrib.layers.fully_connected(F1, 48, activation_fn = tf.nn.relu, reuse = reuse, scope = "FC2")
    F3 = tf.contrib.layers.fully_connected(F2, 16, activation_fn = tf.nn.relu, reuse = reuse, scope = "FC3")
    F4 = tf.contrib.layers.fully_connected(F3, 4, activation_fn = tf.nn.relu, reuse = reuse, scope = "FC4")
    Z = tf.contrib.layers.fully_connected(F4, 1, activation_fn = None, reuse = reuse, scope = "FC5")

    return Z
    
def compute_cost(Z, Y):
    
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = Y, logits = Z))

    return cost

m, n_H, n_W, n_C = X_train.shape
n_Y = Y_train.shape[1]
#costs = []

tf.reset_default_graph()

X, Y = create_placeholders(n_H, n_W, n_C, n_Y)
parameters = initialize_parameters(n_H, n_W, n_C, n_Y)
Z = forward_propagation(X, parameters, n_H, n_W, n_C, n_Y)
cost = compute_cost(Z, Y)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.000025).minimize(cost)

session = tf.Session()
session.run(tf.global_variables_initializer())

for i in range(35):
    minibatch_cost = 0.
    num_batches = int(m / 64)
    minibatches = random_mini_batches(X_train, Y_train, mini_batch_size= 64)

    for minibatch in minibatches:
        (minibatch_X, minibatch_Y) = minibatch
        _, temp_cost = session.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
        minibatch_cost += temp_cost
    
    print("Cost after %d epochs = %f" %(i+1, minibatch_cost))
#    if i%5 == 0:
#        costs.append(minibatch_cost)

parameters = session.run(parameters)

del minibatches, minibatch, minibatch_X, minibatch_Y, i, minibatch_cost, temp_cost, num_batches
#del X_test, Y_test, X_train, Y_train

######################### CALL THESE FOR DATASET  #############################
X_train = np.load("npy_files/X_train.npy")
X_test = np.load("npy_files/X_test.npy")
Y_train = np.load("npy_files/Y_train.npy")
Y_test = np.load("npy_files/Y_test.npy")
############################################################################### 

del X_train, Y_train
del X_test, Y_test

#X_test
X_test = np.load("npy_files/X_test.npy")

minibatches_X_test = mini_batches_X(X_test, mini_batch_size = 128)
del X_test

l = []
count = 0
for i in minibatches_X_test:
    Y_pred_test_i = session.run(forward_propagation(i, parameters, n_H, n_W, n_C, n_Y, reuse = True))
    Y_pred_test_i = session.run(tf.nn.sigmoid(Y_pred_test_i))
    Y_pred_test_i = np.array([1 if(i>0.5) else 0 for i in Y_pred_test_i]).reshape((Y_pred_test_i.shape[0], 1))
    print(count)
    count+=1
    l.append(Y_pred_test_i)
    

Y_pred_test = l[0]
for i in range(1, len(l)):
    Y_pred_test = np.concatenate((Y_pred_test, l[i]), axis = 0)

del Y_pred_test_i, count, i, l, minibatches_X_test

#Finding results
Y_test = np.load("npy_files/Y_test.npy")
count_test = 0
for i in range(3558):
    if(Y_test[i] == Y_pred_test[i]): count_test +=1


###############################################################################
print("Test Accuracy: %.4f" %(count_test/len(Y_test)))
###############################################################################

del Y_test, count_test

#X_train
X_train = np.load('npy_files/X_train.npy')
minibatches_X_train = mini_batches_X(X_train, mini_batch_size = 256)
del X_train

l = []
count = 0
for i in minibatches_X_train:
    Y_pred_train_i = session.run(forward_propagation(i, parameters, n_H, n_W, n_C, n_Y, reuse = True))
    Y_pred_train_i = session.run(tf.nn.sigmoid(Y_pred_train_i))
    Y_pred_train_i = np.array([1 if i>0.5 else 0 for i in Y_pred_train_i]).reshape((Y_pred_train_i.shape[0], 1))
    print(count)
    count+=1
    l.append(Y_pred_train_i)

Y_pred_train = l[0]
for i in range(1, len(l)):
    Y_pred_train = np.concatenate((Y_pred_train, l[i]), axis = 0)

del Y_pred_train_i, count, i, l, minibatches_X_train

#Finding train results
Y_train = np.load("npy_files/Y_train.npy")

cm = 0
for i in range(len(Y_train)):
    if Y_train[i][0] == Y_pred_train[i][0]: cm+=1

###############################################################################
print("Test Accuracy: %.4f" %(cm/len(Y_train)))
###############################################################################

del Y_pred_train, Y_train, cm, i, m, n_C, n_W, n_H, n_Y, parameters

#tf.saved_model.simple_save(session, "tfmodel", inputs = None, outputs=None)

