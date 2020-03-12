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
h, w = 0, 0
for i in parasitized:
    x = cv2.imread('cell_images\\Parasitized\\'+i).shape
    h+=x[0]
    w+=x[1]    
    count+=1
    if(count%1000==0): print(count)
for i in uninfected:
    x = cv2.imread('cell_images\\Uninfected\\'+i).shape
    h+=x[0]
    w+=x[1]    
    count+=1
    if(count%1000==0): print(count)
    
# h/(13779*2)
# w/(13779*2)


del count, h, i, w, x
#Result
#Average file size will be (h, w) = (128, 128)

#Creating image dataset with data augmentation
count = 0
uninfected_image_array, parasitized_image_array = [], []
for i in parasitized:
    img = Image.open('cell_images\\Parasitized\\'+i)
    img = img.resize((128, 128))
#    rotate_45 = np.array(img.rotate(45))
#    rotate_75 = np.array(img.rotate(75))
#    blur = cv2.blur(np.array(img), (10, 10))
    img = np.array(img)
    parasitized_image_array+=[img] #[blur, rotate_45, rotate_75] #
    count+=1
    if(count%1000==0): print(count)
for i in uninfected:
    img = Image.open('cell_images\\Uninfected\\'+i)
    img = img.resize((128, 128))
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

###############################################################################
#                         TENSORFLOW - MODEL                                  #

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
    permutation = list(np.random.permutation(m))
    shuffled_X, shuffled_Y = X[permutation,:,:,:], Y[permutation,:]

    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
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

    X = tf.placeholder(dtype = tf.float32, shape = (None, n_H, n_W, n_C), name = "X")
    Y = tf.placeholder(dtype = tf.float32, shape = (None, n_Y), name = "Y")

    return X, Y

def convolutional_layer(inputs, shape, strides, padding, name = 'conv_layer_0'):
    w = tf.Variable(tf.truncated_normal(shape, stddev = 0.01), name = name+'_w')  #truncated_normal will generate random valued tensor 
                                                                            #of specified shape, 
                                                                            #try initializing values to be random or 0 or 1
    b = tf.Variable(tf.constant(0.01, shape = [shape[3]]), name = name+'_b')      #initializing b = tensor with constant value of 0.01, try 1? 
    
    conv = tf.nn.conv2d(inputs, w, strides = [1, strides, strides, 1], padding = padding)   #check for name?

    conv = tf.add(conv, b, name = name)
    
    #try including batch normalization????
#    if batch_norm:
#        depth = shape[3]
#        scale = tf.Variable(tf.ones([depth, ], dtype='float32'), name='scale')
#        shift = tf.Variable(tf.zeros([depth, ], dtype='float32'), name='shift')
#        mean = tf.Variable(tf.ones([depth, ], dtype='float32'), name='rolling_mean')
#        variance = tf.Variable(tf.ones([depth, ], dtype='float32'), name='rolling_variance')
#
#        conv_bn = tf.nn.batch_normalization(conv, mean, variance, shift, scale, 1e-05)
#        conv = tf.add(conv_bn, biases)
#        conv = tf.maximum(self.alpha * conv, conv)
#    else:
#        conv = tf.add(conv, biases)
    
    return conv


def pooling_layer(inputs, ksize, strides, padding, name = 'pool_0'):  #try adding ksize and strides as parameters to fn?
    pool = tf.nn.max_pool(inputs, ksize = [1, ksize, ksize, 1], strides = [1, strides, strides, 1], padding = padding, name = name)
    
    return pool

# inputs = X
# tf.reset_default_graph()
# X, Y = create_placeholders(n_H, n_W, n_C, n_Y)

def model(inputs, reuse = None):
    op1 = convolutional_layer(inputs, [20, 20, 3, 96], 2, "VALID", 'conv_layer_1')
    op2 = tf.nn.relu(op1, name = 'relu_layer_2')
    
    op3 = pooling_layer(op2, 3, 2, "VALID", 'max_pool_layer_3')
    op4 = tf.nn.relu(op3, name = 'relu_layer_4')
    
    op5 = convolutional_layer(op4, [5, 5, 96, 256], 1, "SAME", 'conv_layer_5')
    op6 = tf.nn.relu(op5, name = 'relu_layer_6')
    
    op7 = pooling_layer(op6, 3, 2, "VALID", 'max_pool_layer_7')
    op8 = tf.nn.relu(op7, name = 'relu_layer_8')
    
    op9 = convolutional_layer(op8, [3, 3, 256, 384], 1, "SAME", 'conv_layer_9')
    op10 = tf.nn.relu(op9, name = 'relu_layer_10')
    
    op11 = convolutional_layer(op10, [3, 3, 384, 384], 1, "SAME", 'conv_layer_11')
    op12 = tf.nn.relu(op11, name = 'relu_layer_12')

    op13 = convolutional_layer(op12, [3, 3, 384, 256], 1, "SAME", 'conv_layer_13')
    op14 = tf.nn.relu(op13, name = 'relu_layer_14')
    
    op15 = pooling_layer(op14, 3, 2, "VALID", 'max_pool_layer_15')
    op16 = tf.nn.relu(op15, name = "relu_layer_16")
    
    op17 = tf.contrib.layers.flatten(op16)
    
    op18 = tf.contrib.layers.fully_connected(op17, 2048, activation_fn = tf.nn.relu, reuse = reuse, scope = "fully_connected_layer_18")

    op19 = tf.contrib.layers.fully_connected(op18, 256, activation_fn = tf.nn.relu, reuse = reuse, scope = "fully_connected_layer_19")

    op20 = tf.contrib.layers.fully_connected(op19, 64, activation_fn = tf.nn.relu, reuse = reuse, scope = "fully_connected_layer_20")

    op21 = tf.contrib.layers.fully_connected(op20, 8, activation_fn = tf.nn.relu, reuse = reuse, scope = "fully_connected_layer_21")

    Z = tf.contrib.layers.fully_connected(op21, 1, activation_fn = None, reuse = reuse, scope = "fully_connected_layer_22")
    
    return Z

def compute_cost(Z, Y):
    
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = Y, logits = Z))

    return cost

m, n_H, n_W, n_C = X_train.shape
n_Y = Y_train.shape[1]
#costs = []

tf.reset_default_graph()

X, Y = create_placeholders(n_H, n_W, n_C, n_Y)
Z = model(X, reuse = None)
cost = compute_cost(Z, Y)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.000025).minimize(cost)

session = tf.Session()
session.run(tf.global_variables_initializer())
session.run(tf.local_variables_initializer())

costs = []

for i in range(2):
    batch_cost = 0
    minibatches = random_mini_batches(X_train, Y_train, mini_batch_size = 32)
    for minibatch_X, minibatch_Y in minibatches:
        _, minibatch_cost = session.run([optimizer, cost], feed_dict = {X:minibatch_X, Y:minibatch_Y})
        batch_cost += minibatch_cost
    print("Epoch %d: Cost: %.6f"%(i+1, batch_cost))
    costs.append(batch_cost)

del X_train, Y_train, batch_cost, costs, i, m, minibatch_X, minibatch_Y, minibatch_cost, minibatches

X_test = np.load("npy_files/X_test.npy")
Y_test = np.load("npy_files/Y_test.npy")

Y_pred = session.run(model(X_test, reuse = tf.AUTO_REUSE))
Y_pred = session.run(tf.nn.sigmoid(Y_pred))


