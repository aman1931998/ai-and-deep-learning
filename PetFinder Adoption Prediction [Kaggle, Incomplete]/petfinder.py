#Importing the modules
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from scipy import misc, ndimage
import json
import cv2

#Importing dataset
breed_labels = pd.read_csv("breed_labels.csv")
color_labels = pd.read_csv("color_labels.csv")
state_labels = pd.read_csv("state_labels.csv")
train = pd.read_csv(r"train\train.csv")
test = pd.read_csv(r"test\test.csv")

#############################    Y_train    ###################################
Y_train_ = train.pop('AdoptionSpeed').values.reshape((train.shape[0], 1))

##################    Concatenating entire Dataset    #########################
dataset = train.append(test) #train x 14993 + test x 3972
del train, test

#############################    Missing    ###################################
missing = dataset.isna().sum()
del missing

##############################    Name    #####################################
Name = dataset.pop('Name')

Name.isna().sum() #got 1668
Name = Name.fillna("No Name Yet")
Name.isna().sum() #got 0
Name = Name.replace(to_replace = "No Name Yet", value = 0)
Name_ = []
for i in Name:
    if(i==0):
        Name_.append(0)
    else:
        Name_.append(len(i.split(',')))
Name = np.array(Name_).reshape((dataset.shape[0], 1))
del Name_, i
#np.save('npy_files/data/Name.npy', Name)
#Name = np.load('npy_files/data/Name.npy')

###############################    Age    #####################################
Age = dataset['Age'].values.reshape((dataset.shape[0], 1))
Age = StandardScaler().fit_transform(Age)
dataset.pop('Age')
#np.save('npy_files/data/Age.npy', Age)
#Age = np.load('npy_files/data/Age.npy')

###############################    Type    ####################################
Type = dataset['Type']
#Type.value_counts()
Type = LabelEncoder().fit_transform(Type).reshape((Type.shape[0], 1))
dataset.pop('Type')
#np.save('npy_files/data/Type.npy', Type)
#Type = np.load('npy_files/data/Type.npy')

##############################    Gender    ###################################
Gender = dataset['Gender'].values.reshape((dataset.shape[0], 1))
#Gender.value_counts()
Gender = OneHotEncoder().fit_transform(Gender).toarray()[:, 1:]
dataset.pop('Gender')
#np.save('npy_files/data/Gender.npy', Gender)
#Gender = np.load('npy_files/data/Gender.npy')

##########################    Maturity_Size    ################################
MaturitySize = dataset['MaturitySize'].values.reshape((dataset.shape[0], 1))
#MaturitySize.value_counts()
MaturitySize = OneHotEncoder().fit_transform(MaturitySize).toarray()[:, :-1]
dataset.pop('MaturitySize')
#np.save('npy_files/data/MaturitySize.npy', MaturitySize)
#MaturitySize = np.load('npy_files/data/MaturitySize.npy')

############################    FurLength    ##################################
FurLength = dataset['FurLength'].values.reshape((dataset.shape[0], 1))
#FurLength.value_counts()
FurLength = OneHotEncoder().fit_transform(FurLength).toarray()[:, :-1]
dataset.pop('FurLength')
#np.save('npy_files/data/FurLength.npy', FurLength)
#FurLength = np.load('npy_files/data/FurLength.npy')

############################    Vaccinated    #################################
Vaccinated = dataset['Vaccinated'].values.reshape((dataset.shape[0], 1))
#Vaccinated.value_counts()
Vaccinated = OneHotEncoder().fit_transform(Vaccinated).toarray()[:, :-1]
dataset.pop('Vaccinated')
#np.save('npy_files/data/Vaccinated.npy', Vaccinated)
#Vaccinated = np.load('npy_files/data/Vaccinated.npy')

#############################    Dewormed    ##################################
Dewormed = dataset['Dewormed'].values.reshape((dataset.shape[0], 1))
#Dewormed.value_counts()
Dewormed = OneHotEncoder().fit_transform(Dewormed).toarray()[:, :-1]
dataset.pop('Dewormed')
#np.save('npy_files/data/Dewormed.npy', Dewormed )
#Dewormed = np.load('npy_files/data/Dewormed.npy')

###########################    Sterilized    ##################################
Sterilized = dataset['Sterilized'].values.reshape((dataset.shape[0], 1))
#Sterilized.value_counts()
Sterilized = OneHotEncoder().fit_transform(Sterilized).toarray()[:, :-1]
dataset.pop('Sterilized')
#np.save('npy_files/data/Sterilized.npy', Sterilized)
#Sterilized = np.load('npy_files/data/Sterilized.npy')

#############################    Health    ####################################
Health = dataset['Health'].values.reshape((dataset.shape[0], 1))
#Health.value_counts()
Health = OneHotEncoder().fit_transform(Health).toarray()[:, :-1]
dataset.pop('Health')
#np.save('npy_files/data/Health.npy', Health)
#Health = np.load('npy_files/data/Health.npy')

############################    Quantity    ###################################
Quantity = dataset['Quantity'].values.reshape((dataset.shape[0], 1))
#Quantity.value_counts()                                      #to see whether MinMax is reasonable or not?
Quantity = StandardScaler().fit_transform(Quantity)           #Should I apply this? MinMax is useless here
dataset.pop('Quantity')
#np.save('npy_files/data/Quantity.npy', Quantity)
#Quantity = np.load('npy_files/data/Quantity.npy')

###############################    Fee    #####################################
Fee = dataset['Fee'].values.reshape((dataset.shape[0], 1))
Fee = MinMaxScaler().fit_transform(Fee)                       #MinMax seems reasonable
dataset.pop('Fee')
#np.save('npy_files/data/Fee.npy', Fee)
#Fee = np.load('npy_files/data/Fee.npy')

############################    VideoAmt    ###################################
VideoAmt = dataset['VideoAmt'].values.reshape((dataset.shape[0], 1))
VideoAmt = MinMaxScaler().fit_transform(VideoAmt)             #MinMax seems reasonable
dataset.pop('VideoAmt')
#np.save('npy_files/data/VideoAmt.npy', VideoAmt)
#VideoAmt = np.load('npy_files/data/VideoAmt.npy')

############################    PhotoAmt    ###################################
PhotoAmt = dataset['PhotoAmt'].values.reshape((dataset.shape[0], 1))
PhotoAmt = MinMaxScaler().fit_transform(PhotoAmt)             #MinMax seems reasonable
dataset.pop('PhotoAmt')
#np.save('npy_files/data/PhotoAmt.npy', PhotoAmt)
#PhotoAmt = np.load('npy_files/data/PhotoAmt.npy')

##########################    Description    ##################################
dataset.pop('Description')  #Useless for me

#############################    Breed    #####################################
Breed1 = dataset['Breed1']
Breed2 = dataset['Breed2']
breed_labels = breed_labels.values
breed_labels_id = breed_labels[:, 0]
Type1 = []
for i in Breed1:
    flag = 1
    for j in range(len(breed_labels)):
        if breed_labels_id[j] == i: Type1.append(breed_labels[j, 1]); flag = 0
    if flag: Type1.append(0)
Type1 = np.array(Type1).reshape((dataset.shape[0], 1))
Type2 = []
for i in Breed2:
    flag = 1
    for j in range(len(breed_labels)):
        if breed_labels_id[j] == i:
            Type2.append(breed_labels[j, 1])
            flag = 0
    if flag:
        Type2.append(0)
Type2 = np.array(Type2).reshape((dataset.shape[0], 1))

del breed_labels, breed_labels_id, i, flag, j, Breed1, Breed2
dataset.pop('Breed1')
dataset.pop('Breed2')
#np.save('npy_files/data/Type1.npy', Type1)
#Type1 = np.load('npy_files/data/Type1.npy')
#np.save('npy_files/data/Type2.npy', Type2)
#Type2 = np.load('npy_files/data/Type2.npy')

#############################    Color1    ####################################
#OneHot Necessary?
Color1 = dataset['Color1'].values.reshape((dataset.shape[0], 1))
Color1 = OneHotEncoder().fit_transform(Color1).toarray()[:, 1:]
#np.save('npy_files/data/Color1.npy', Color1)
#Color1 = np.load('npy_files/data/Color1.npy')

#############################    Color2    ####################################
#OneHot Necessary?
Color2 = dataset['Color2'].values.reshape((dataset.shape[0], 1))
Color2 = OneHotEncoder().fit_transform(Color2).toarray()[:, 1:]
#np.save('npy_files/data/Color2.npy', Color2)
#Color2 = np.load('npy_files/data/Color2.npy')

#############################    Color3    ####################################
#OneHot Necessary?
Color3 = dataset['Color3'].values.reshape((dataset.shape[0], 1))
Color3 = OneHotEncoder().fit_transform(Color3).toarray()[:, 1:]
#np.save('npy_files/data/Color3.npy', Color3)
#Color3 = np.load('npy_files/data/Color3.npy')

dataset.pop('Color1')
dataset.pop('Color2')
dataset.pop('Color3')
del color_labels

#############################    State    #####################################
State = dataset['State'].values.reshape((dataset.shape[0], 1))
State = LabelEncoder().fit_transform(State).reshape((dataset.shape[0], 1))
dataset.pop('State')
#np.save('npy_files/data/State.npy', State)
#State = np.load('npy_files/data/State.npy')
del state_labels

##########################   Combining data    ################################
dataset1 = np.concatenate((Age, Color1, Color2, Color3, Dewormed, 
                           Fee, FurLength, Gender, Health, MaturitySize, 
                           Name, PhotoAmt, Quantity, State, Sterilized, Type, 
                           Type1, Type2, Vaccinated, VideoAmt), axis = 1)

#np.save('npy_files/dataset1.npy', dataset1)
#dataset1 = np.load('npy_files/dataset1.npy')
del Age, Color1, Color2, Color3, Dewormed, Fee, FurLength, Gender, Health, MaturitySize, Name, PhotoAmt, Quantity, State, Sterilized, Type, Type1, Type2, Vaccinated, VideoAmt

#############################    Pet ID    ####################################
PetID = dataset.pop("PetID").values.reshape((dataset.shape[0], 1))
PetID_train = PetID[:14993]
PetID_test = PetID[14993:]

#List of images
images_train = list(os.walk('train_images'))[0][2]
images_test = list(os.walk('test_images'))[0][2]

train_images = []
train_dataset2 = []
Y_train = []

count = 0

for i in range(len(PetID_train)):
    val = PetID_train[i][0]
    for j in range(1, 100):
        image_path = os.path.join('train_images', val) + '-' + str(j) + '.jpg'
        img = cv2.imread(image_path)
        if img is None: break
        img = cv2.resize(img, (112, 112)) / 255.0 * 2.0 - 1.0    #Mean norm? or std Norm?
        img = np.array(img, dtype = np.float32)
        train_dataset2.append(dataset1[i])
        train_images.append(img)
        Y_train.append(Y_train_[i])
        if count % 50 == 0: print(count)
        count+=1

train_dataset2 = np.array(train_dataset2)
train_images = np.array(train_images)
Y_train = np.array(Y_train).reshape((len(Y_train), 1))
np.save('npy_files/train_images.npy', train_images)   #
np.save('npy_files/train_data.npy', train_dataset2)   #
np.save('npy_files/Y_train.npy', Y_train)             #
#del train_dataset2, train_images
del val, i, j, image_path, dataset, PetID_train, PetID, Y_train_

test_images = []
test_dataset2 = []
for i in range(len(PetID_test)):
    val = PetID_test[i][0]
    for j in range(1, 100):
        image_path = os.path.join('test_images', val) + '-' + str(j) + '.jpg'
        img = cv2.imread(image_path)
        if img is None: break
        img = cv2.resize(img, (112, 112)) / 255.0 * 2.0 - 1.0    #Mean norm? or std Norm?
        img = np.array(img, dtype = np.float32)
        test_dataset2.append(dataset1[i+14993])
        test_images.append(img)

np.save('npy_files/test_images.npy', test_images)   #
np.save('npy_files/test_data.npy', test_dataset2)   #
del test_images, test_dataset2
del PetID_test, dataset1, i, image_path, images_train, images_test, j, val

#train_dataset2 = np.load('npy_files/train_data.npy')      #(None, 42)
#train_images = np.load('npy_files/train_images.npy')  #(None, 224, 224, 3)
#np.save('npy_files/train_images.npy', train_images)
#np.save('npy_files/train_data.npy', train_dataset2)

#Y_train = np.load('npy_files/Y_train.npy')

def random_mini_batches(X_data, X_images, Y, mini_batch_size = 64):
    m = X_data.shape[0]                  # number of training examples
    mini_batches = []
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X_data = X_data[permutation,:]
    shuffled_X_images = X_images[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X_data = shuffled_X_data[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch_X_images = shuffled_X_images[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X_data, mini_batch_X_images, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X_data = shuffled_X_data[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch_X_images = shuffled_X_images[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X_data, mini_batch_X_images, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

#############################    MODEL    #####################################
def create_placeholders(n_H, n_W, n_C, n_Y, n_D):
    X_data = tf.placeholder(dtype = tf.float32, shape = (None, n_D), name = "X_data")
    X_images = tf.placeholder(dtype = tf.float32, shape = (None, n_H, n_W, n_C), name = "X_images")
    Y = tf.placeholder(dtype = tf.float32, shape = (None, n_Y), name = "Y")
    
    return X_data, X_images, Y

def initialize_parameters_1(n_D, n_Y):
    
    parameters = {}
    
    W01 = tf.get_variable(name = "W01", shape = (n_D, 40), dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 32))
    W02 = tf.get_variable(name = "W02", shape = (40, 32), dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 32))
    W03 = tf.get_variable(name = "W03", shape = (32, 40), dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 32))
    W04 = tf.get_variable(name = "W04", shape = (40, 16), dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 32))
    W05 = tf.get_variable(name = "W05", shape = (16, 8), dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 32))
    B01 = tf.get_variable(name = "B01", shape = (1, 40), dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 32))
    B02 = tf.get_variable(name = "B02", shape = (1, 32), dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 32))
    B03 = tf.get_variable(name = "B03", shape = (1, 40), dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 32))
    B04 = tf.get_variable(name = "B04", shape = (1, 16), dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 32))
    B05 = tf.get_variable(name = "B05", shape = (1, 8), dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 32))
    
    parameters["W01"] = W01
    parameters["W02"] = W02
    parameters["W03"] = W03
    parameters["W04"] = W04
    parameters["W05"] = W05
    parameters["B01"] = B01
    parameters["B02"] = B02
    parameters["B03"] = B03
    parameters["B04"] = B04
    parameters["B05"] = B05
    
    return parameters

def convolutional_layer(inputs, shape, name = 'conv_layer_0', padding = "SAME"):
    w = tf.Variable(tf.truncated_normal(shape, stddev = 0.01), name = 'w')  #truncated_normal will generate random valued tensor 
                                                                            #of specified shape, 
                                                                            #try initializing values to be random or 0 or 1
    b = tf.Variable(tf.constant(0.01, shape = [shape[3]]), name = 'b')      #initializing b = tensor with constant value of 0.01, try 1? 
    
    conv = tf.nn.conv2d(inputs, w, strides = [1, 1, 1, 1], padding = padding, name = name)   #check for name?

    conv = tf.add(conv, b)
    
    return conv

def pooling_layer(inputs, name = 'pool_0'):  #try adding ksize and strides as parameters to fn?
    pool = tf.nn.max_pool(inputs, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME", name = name)
    
    return pool

def model(X_data, X_images, parameters, reuse = None):
    W01 = parameters['W01']
    W02 = parameters['W02']
    W03 = parameters['W03']
    W04 = parameters['W04']
    W05 = parameters['W05']
    B01 = parameters['B01']
    B02 = parameters['B02']
    B03 = parameters['B03']
    B04 = parameters['B04']
    B05 = parameters['B05']

    
    Z1 = tf.add(tf.matmul(X_data, W01), B01, name = "Z1")
    A1 = tf.nn.relu(Z1, name = "A1")
    Z2 = tf.add(tf.matmul(A1, W02), B02, name = "Z2")
    A2 = tf.nn.relu(Z2, name = "A2")
    Z3 = tf.add(tf.matmul(A2, W03), B03, name = "Z3")
    A3 = tf.nn.relu(Z3, name = "A3")
    Z4 = tf.add(tf.matmul(A3, W04), B04, name = "Z4")
    A4 = tf.nn.relu(Z4, name = "A4")
    Z5 = tf.add(tf.matmul(A4, W05), B05, name = "Z5")
    A5 = tf.nn.relu(Z5, name = "A5")   #Output of data part is (?, 8)
    
    op = convolutional_layer(X_images, [3, 3, 3, 32], name = 'conv_0')
    op = pooling_layer(op, name = 'pool_1')

    op = convolutional_layer(op, [3, 3, 32, 64], name = 'conv_2')
    op = pooling_layer(op, name = 'pool_3')

    op = convolutional_layer(op, [3, 3, 64, 128], name = 'conv_4')
    op = convolutional_layer(op, [1, 1, 128, 64], name = 'conv_5')
    op = convolutional_layer(op, [3, 3, 64, 128], name = 'conv_6')
    op = pooling_layer(op, name = 'pool_7')

    op = convolutional_layer(op, [3, 3, 128, 256], name = 'conv_8')
    op = convolutional_layer(op, [1, 1, 256, 128], name = 'conv_9')
    op = convolutional_layer(op, [3, 3, 128, 256], name = 'conv_10')
#    op = pooling_layer(op, name = 'pool_11')

    op = convolutional_layer(op, [3, 3, 256, 512], name = 'conv_12')
    op = convolutional_layer(op, [1, 1, 512, 256], name = 'conv_13')
    op = convolutional_layer(op, [3, 3, 256, 512], name = 'conv_14')
    op = convolutional_layer(op, [1, 1, 512, 256], name = 'conv_15')
    op = convolutional_layer(op, [3, 3, 256, 512], name = 'conv_16')
    op = pooling_layer(op, name = 'pool_17')

    op = convolutional_layer(op, [3, 3, 512, 1024], name = 'conv_18')
    op = convolutional_layer(op, [1, 1, 1024, 512], name = 'conv_19')
    op = convolutional_layer(op, [3, 3, 512, 1024], name = 'conv_20')
    op = convolutional_layer(op, [1, 1, 1024, 512], name = 'conv_21')
    op = convolutional_layer(op, [3, 3, 512, 1024], name = 'conv_22')
    
    op = convolutional_layer(op, [3, 3, 1024, 1024], name = 'conv_23')
    op = convolutional_layer(op, [3, 3, 1024, 1024], name = 'conv_24')

    op = convolutional_layer(op, [1, 1, 1024, 512], name = 'conv_25')  #?

    op = convolutional_layer(op, [3, 3, 512, 1024], name = 'conv_26')     
    op = convolutional_layer(op, [7, 7, 1024, 392], name = 'conv_27', padding = "VALID")
    op = tf.layers.flatten(op)
    op = tf.concat((op, A5), axis=1)
    op = tf.contrib.layers.fully_connected(op, num_outputs = 160, activation_fn = tf.nn.relu, reuse = reuse, scope = "F1")
    op = tf.contrib.layers.fully_connected(op, num_outputs = 64, activation_fn = tf.nn.relu, reuse = reuse, scope = "F2")
    op = tf.contrib.layers.fully_connected(op, num_outputs = 16, activation_fn = tf.nn.relu, reuse = reuse, scope = "F3")
    op = tf.contrib.layers.fully_connected(op, num_outputs = 1, activation_fn = None, reuse = reuse, scope = "F4")
    
    return op

def loss_function(logits, labels):  #None to batch size?    
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))
    return loss

tf.reset_default_graph()


m, n_H, n_W, n_C = train_images.shape
n_D = train_dataset2.shape[1]
n_Y = Y_train.shape[1]

X_data, X_images, Y = create_placeholders(n_H, n_W, n_C, n_Y, n_D)
parameters = initialize_parameters_1(n_D, n_Y)
Z = model(X_data, X_images, parameters)
loss = loss_function(logits = Z, labels = Y)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.000025).minimize(loss)

session = tf.Session()
session.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter("D:\\Datsets\\petfinder-adoption-prediction\\tmp\\tensorflowlogs",session.graph)
writer.close()

for i in range(3):
    minibatch_cost = 0.
    num_batches = int(58311//32)
    minibatches = random_mini_batches(train_dataset2, train_images, Y_train, mini_batch_size = 32)
    count_ = 0
    for minibatch in minibatches:
        (mini_batch_X_data, mini_batch_X_images, mini_batch_Y) = minibatch
        _, temp_cost = session.run([optimizer, loss], feed_dict={X_data:mini_batch_X_data, X_images:mini_batch_X_images, Y:mini_batch_Y})
        minibatch_cost += temp_cost
        if not count_%10: print("minibatch %d" %count_, minibatch_cost)
        count_+=1
    print("Cost after %d epochs = %f" %(i+1, minibatch_cost))
#    if i%5 == 0:
#        costs.append(minibatch_cost)




#defining iou
#defining overlap mech? (name wtf)









