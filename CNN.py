import tensorflow as tf
import os
from PIL import Image
import numpy as np
import math
from sklearn.utils import shuffle

# define one hot method for output
def one_hot(i):
    a = np.zeros(51, 'float32')
    a[i] = 1
    return a

# get all folders paths
all_folders = os.listdir("test")

# define global variables for train, validation and test sets
train_labels = []
train_images = []
validation_images = []
validation_labels = []
test_images = []
test_labels = []

n_input = 784
n_output = 51
image_size = [28, 28]
batch_size = 50
batch_counter = 0
train_size = 0.6
validation_size = 0.2
test_size = 0.2
'''
Basic steps:
1. import images
2. convert all images
3. build CNN
4. train the model on train & validation set
5. test the model on test set
'''

for folder in all_folders:
    files = os.listdir("test/" + folder)
    folder_images = []#only the images of one folder
    folder_labels = []
    for file in files:
        image = Image.open("test/" + folder + "/" + file).convert('L')  # converting to grayscale
        image = image.resize(image_size, Image.ANTIALIAS) #resizing the picture
        image_to_array = np.array(image)
        image_to_array = np.reshape(image_to_array,[784])
        image_to_array = image_to_array / 255 # normalization pixles
        folder_images.append(image_to_array)
        folder_labels.append(one_hot(int(folder)))

    train = math.floor(train_size * len(folder_images))#size of train in one folder
    test = math.floor(test_size * len(folder_images))
    validation = math.floor(validation_size * len(folder_images))

    train_images.extend(folder_images[:train])
    train_labels.extend(folder_labels[:train])

    validation_images.extend(folder_images[train:train+validation])
    validation_labels.extend(folder_labels[train:train+validation])

    test_images.extend(folder_images[train + test:])
    test_labels.extend(folder_labels[train + test:])


def get_next_batch(num):
    batch_images = train_images[num: num + batch_size]
    batch_labels = train_labels[num: num + batch_size]
    return batch_images, batch_labels


x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])
x_tensor = tf.reshape(x, [-1, 28, 28, 1])

# functions for convolutions and pooling
def conv_2d(x,w):
    # padding-if the image sizes stays the same
  return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # ksize the size of the window
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def weights(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def biases(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Weight matrix is [height x width x input x output]
filter_size = 5
n_filters_1 = 16
n_filters_2 = 16

w_conv1 = weights([filter_size, filter_size, 1, n_filters_1])
b_conv1 = biases([n_filters_1])
w_conv2 = weights([filter_size, filter_size, n_filters_1, n_filters_2])
b_conv2 = biases([n_filters_2])

# the layers
h_conv1 = tf.nn.relu(conv_2d(x_tensor, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
h_conv2 = tf.nn.relu(conv_2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# (28x28 -> 14x14 -> 7x7)
h_conv2_reshape = tf.reshape(h_pool2, [-1, 7 * 7 * n_filters_2])

# FC layer
n_fc = 1024
w_fc1 = weights([7 * 7 * n_filters_2, n_fc])
b_fc1 = biases([n_fc])
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_reshape, w_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weights([n_fc, n_output])
b_fc2 = biases([n_output])
y_pred = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(y_pred, y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# variables:
sess = tf.Session()
sess.run(tf.global_variables_initializer())

n_epochs = 5
loss_list = list()

for epoch_i in range(n_epochs):

    # shuffle the train labels and train images toghther
    train_images, train_labels = shuffle(train_images, train_labels)

    for i in range(0, len(train_images), batch_size):
        batch_xs, batch_ys = get_next_batch(i)
        sess.run(optimizer, feed_dict={
            x: batch_xs, y: batch_ys, keep_prob: 0.5})

    loss = sess.run(accuracy, feed_dict={
                       x: validation_images,
                       y: validation_labels,
                       keep_prob: 1.0 })
    print('Validation accuracy for epoch {} is: {}'.format(epoch_i + 1, loss))
    loss_list.append(loss)



print("Accuracy for test set: {}".format(sess.run(accuracy,
               feed_dict={
                   x: test_images,
                   y: test_labels,
                   keep_prob: 1.0
               })))

saver = tf.train.Saver()
save_path = saver.save(sess, "model")
print("Model is saved in: ",  save_path)