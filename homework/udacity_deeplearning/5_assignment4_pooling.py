import numpy as np
import tensorflow as tf
import pickle
from os.path import join, expanduser

import matplotlib.pyplot as plt
import random

mnist_root      = join(expanduser("~"), 'mldata')
data_root       = join(mnist_root, 'notmnist')
pickle_file     = join(data_root, "notMNIST_unique.pickle")

img_size = 28
num_labels = 10

batch_size = 16
patch_size = 5
depth=16
num_hidden=64

num_channels = 1  # grayscale

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, img_size, img_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels.reshape(-1, 1)).astype(np.float32)
    return dataset, labels

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

graph = tf.Graph()

with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, img_size, img_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variable
    W1 = tf.Variable(tf.truncated_normal(shape=[patch_size, patch_size, num_channels, depth], stddev=0.1))
    B1 = tf.Variable(tf.zeros([depth]))
    W2 = tf.Variable(tf.truncated_normal(shape=[patch_size, patch_size, depth, depth], stddev=0.1))
    B2 = tf.Variable(tf.constant(1.0, shape=[depth]))
    W3 = tf.Variable(tf.truncated_normal(shape=[img_size // 4 * img_size // 4 * depth, num_hidden], stddev=0.1))
    B3 = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    W4 = tf.Variable(tf.truncated_normal(shape=[num_hidden, num_labels], stddev=0.1))
    B4 = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    # Model
    def model(data):
        conv = tf.nn.conv2d(data, W1, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + B1)
        print(conv.get_shape().as_list())
        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv = tf.nn.conv2d(pool, W2, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + B2)
        print(conv.get_shape().as_list())
        pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        shape = pool.get_shape().as_list()
        print(shape)
        reshape = tf.reshape(pool, [shape[0], shape[1]*shape[2]*shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, W3) + B3)
        return tf.matmul(hidden, W4) + B4

    # Training Computation
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)
    )

    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # predictions for the training, validation and test data
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction  = tf.nn.softmax(model(tf_test_dataset))

num_steps = 8001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset : (offset+batch_size), :,:,:]
        batch_labels = train_labels[offset: (offset+batch_size), :]
        feed_dict = {tf_train_dataset:batch_data, tf_train_labels:batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict
        )
        if (step % 500 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(
                valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))





'''
Some Notes:

In [25]: x = tf.constant([[1,2,3], [4,5,6]])

In [26]: x.get_shape()
Out[26]: TensorShape([Dimension(2), Dimension(3)])

In [27]: x.get_shape().as_list()
Out[27]: [2, 3]
'''

"""
running result:

Training set (274491, 28, 28) (274491,)
Validation set (15249, 28, 28) (15249,)
Test set (14750, 28, 28) (14750,)
Training set (274491, 28, 28, 1) (274491, 10)
Validation set (15249, 28, 28, 1) (15249, 10)
Test set (14750, 28, 28, 1) (14750, 10)
[16, 28, 28, 16]
[16, 14, 14, 16]
[16, 7, 7, 16]
[15249, 28, 28, 16]
[15249, 14, 14, 16]
[15249, 7, 7, 16]
[14750, 28, 28, 16]
[14750, 14, 14, 16]
[14750, 7, 7, 16]
2017-09-25 16:41:19.129810: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-09-25 16:41:19.129830: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
Initialized
Minibatch loss at step 0: 3.949861
Minibatch accuracy: 12.5%
Validation accuracy: 10.0%
Minibatch loss at step 500: 1.095544
Minibatch accuracy: 75.0%
Validation accuracy: 79.8%
Minibatch loss at step 1000: 0.315458
Minibatch accuracy: 93.8%
Validation accuracy: 81.5%
Minibatch loss at step 1500: 0.339049
Minibatch accuracy: 87.5%
Validation accuracy: 84.1%
Minibatch loss at step 2000: 0.379893
Minibatch accuracy: 93.8%
Validation accuracy: 85.2%
Minibatch loss at step 2500: 0.243178
Minibatch accuracy: 87.5%
Validation accuracy: 85.2%
Minibatch loss at step 3000: 0.348086
Minibatch accuracy: 81.2%
Validation accuracy: 85.2%
Minibatch loss at step 3500: 0.490458
Minibatch accuracy: 81.2%
Validation accuracy: 85.8%
Minibatch loss at step 4000: 0.761396
Minibatch accuracy: 75.0%
Validation accuracy: 86.4%
Minibatch loss at step 4500: 0.005630
Minibatch accuracy: 100.0%
Validation accuracy: 86.4%
Minibatch loss at step 5000: 0.389644
Minibatch accuracy: 87.5%
Validation accuracy: 87.0%
Minibatch loss at step 5500: 0.412554
Minibatch accuracy: 93.8%
Validation accuracy: 87.0%
Minibatch loss at step 6000: 0.415029
Minibatch accuracy: 81.2%
Validation accuracy: 87.1%
Minibatch loss at step 6500: 0.467780
Minibatch accuracy: 87.5%
Validation accuracy: 87.3%
Minibatch loss at step 7000: 0.373975
Minibatch accuracy: 87.5%
Validation accuracy: 87.4%
Minibatch loss at step 7500: 0.477503
Minibatch accuracy: 87.5%
Validation accuracy: 87.3%
Minibatch loss at step 8000: 0.417668
Minibatch accuracy: 87.5%
Validation accuracy: 87.9%
Test accuracy: 93.7%
"""









