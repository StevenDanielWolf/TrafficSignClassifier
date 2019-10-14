##Import Statements##
import matplotlib.pyplot as plt
import pickle
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from sklearn.utils import shuffle
import glob
import cv2
import csv
import random


# Load pickled data
training_file = "./data/train.p"
validation_file="./data/valid.p"
testing_file = "./data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# Load label description from .csv file
signnames = np.genfromtxt('signnames.csv', skip_header=1, dtype=[('myint','i8'), ('mysring','S55')], delimiter=',')


#Basic data summary
n_train = len(X_train)

n_validation = len(X_valid)

n_test = len(X_test)

image_shape = X_train[0].shape

n_classes = len(np.unique(y_train))


#Preprocessing
def preprocessing(img):
    # normalizes image data and converts to gray scale
    return (np.sum(img/3, axis=3, keepdims=True) - 128)/128

X_train = preprocessing(X_train)
X_test = preprocessing(X_test)
X_valid = preprocessing(X_valid)

#Model Architecture

##HYPERPARAMETERS##
EPOCHS = 25
BATCH_SIZE = 128


def conv2d(x, W, b, strides = 1):
    # Convolutional layer
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return x

def maxpool2d(x, k=2):
    # Max pooling layer
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='VALID')

def LeNet(x):
    # Architecture based on LeNet

    # Initialize weights with normal distribution
    mu = 0
    sigma = 0.1

    weights = {
    'wc1': tf.Variable(tf.truncated_normal([5, 5, 1, 6], mu, sigma)),
    'wc2': tf.Variable(tf.truncated_normal([5, 5, 6, 16], mu, sigma)),
    'wd1': tf.Variable(tf.truncated_normal([400, 120], mu, sigma)),
    'wd2': tf.Variable(tf.truncated_normal([120, 84], mu, sigma)),
    'out': tf.Variable(tf.truncated_normal([84, 43], mu, sigma))}

    # Initialize zero biases
    biases = {
    'bc1': tf.Variable(tf.zeros(6)),
    'bc2': tf.Variable(tf.zeros(16)),
    'bd1': tf.Variable(tf.zeros(120)),
    'bd2': tf.Variable(tf.zeros(84)),
    'out': tf.Variable(tf.zeros(43))}

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = maxpool2d(conv1, k=2)

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = maxpool2d(conv2, k=2)

    # Flatten. Input = 5x5x16. Output = 400.
    x = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    x = tf.add(tf.matmul(x, weights['wd1']), biases['bd1'])

    # Activation.
    x = tf.nn.relu(x)

    # Dropout Layer
    x = tf.nn.dropout(x, keep_prob)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    x = tf.add(tf.matmul(x, weights['wd2']), biases['bd2'])

    # Activation.
    x = tf.nn.relu(x)

    #Dropout Layer
    x = tf.nn.dropout(x, keep_prob)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    logits = tf.add(tf.matmul(x, weights['out']), biases['out'])

    return logits



x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32) # probability to keep units
one_hot_y = tf.one_hot(y, 43) # convert labels to one_hot encoding


# Learning rate
rate = 0.0009

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data):
    # Evaluates accuracy
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples



saver = tf.train.Saver()


#TensorFlow Session
with tf.Session() as sess:
    # Training Session
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './lenet') # Save model
    print("Model saved")



# Evaluate the accuracy of the model on test dataset. Run only once!

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     saver2 = tf.train.import_meta_graph('./lenet.meta')
#     saver2.restore(sess, "./lenet")
#     test_accuracy = evaluate(X_test, y_test)
#     print("Test Set Accuracy = {:.3f}".format(test_accuracy))


###Test images###

# example_images = sorted(glob.glob('./ExampleTrafficSigns/*.jpg'))
# example_labels = np.array([11, 25, 13, 16, 17])
#
# signs = []
# labels = {}
# plots = {}
#
# # PLot example images found online
# for i, image in enumerate(example_images):
#     img = cv2.imread(image)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img,(32,32))
#     signs.append(img)
#     plots[i] = img
#     labels[i] = signnames[example_labels[i]][1].decode('ascii')
#
#
# # Normalize example images and convert to gray scale
# signs = np.asarray(signs)
# signs_gray = np.sum(signs/3, axis=3, keepdims=True)
# signs_normalized = (signs_gray - 128)/128
#
#
# with tf.Session() as sess:
#     # Run LeNet on example images and compute accuracy
#     sess.run(tf.global_variables_initializer())
#     saver3 = tf.train.import_meta_graph('./lenet.meta')
#     saver3.restore(sess, "./lenet")
#     accuracy = evaluate(signs_normalized, example_labels)
#
#
# print("The model predicts the five example images with {:.0f}% accuracy".format(100*accuracy))'
