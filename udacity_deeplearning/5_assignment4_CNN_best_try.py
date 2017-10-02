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
batch_size = 128
num_channels = 1 # grayscale
patch_size = 5

input_dim = (batch_size, img_size, img_size, num_channels)
output_dim = (batch_size, num_labels)
depth1 = 6
depth2 = 16
patch1 = [patch_size, patch_size, num_channels, depth1]
patch2 = [patch_size, patch_size, depth1, depth2]
fc1_hidden = 120
fc2_hidden = 84
pooling_kernel_size = [1, 2, 2, 1]
patch_stride = [1, 1, 1, 1]
pooling_stride = [1, 2, 2, 1]

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
    tf_train_dataset = tf.placeholder(tf.float32, shape=input_dim)
    tf_train_labels = tf.placeholder(tf.float32, shape=output_dim)
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)



    # Variable
    W1 = tf.Variable(tf.truncated_normal(shape=patch1, stddev=0.1))
    B1 = tf.Variable(tf.zeros([depth1]))
    W2 = tf.Variable(tf.truncated_normal(shape=patch2, stddev=0.1))
    B2 = tf.Variable(tf.constant(1.0, shape=[depth2]))
    W3 = tf.Variable(tf.truncated_normal(shape=[img_size // 4 * img_size // 4 * depth2, fc1_hidden], stddev=0.1))
    B3 = tf.Variable(tf.constant(1.0, shape=[fc1_hidden]))
    W4 = tf.Variable(tf.truncated_normal(shape = [fc1_hidden, fc2_hidden], stddev=0.1))
    B4 = tf.Variable(tf.constant(1.0, shape=[fc2_hidden]))
    W5 = tf.Variable(tf.truncated_normal(shape=[fc2_hidden, num_labels], stddev=0.1))
    B5 = tf.Variable(tf.constant(1.0, shape=[num_labels]))
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.05, global_step, 1000, 0.98, staircase=True)


    # Model
    def model(data):
        conv = tf.nn.conv2d(data, W1, patch_stride, padding='SAME')
        hidden = tf.nn.relu(conv + B1)
        pool = tf.nn.max_pool(hidden, ksize=pooling_kernel_size, strides=pooling_stride, padding='SAME')
        conv = tf.nn.conv2d(pool, W2, patch_stride, padding='SAME')
        hidden = tf.nn.relu(conv + B2)
        pool = tf.nn.max_pool(hidden, ksize=pooling_kernel_size, strides=pooling_stride, padding='SAME')
        shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool, [shape[0], shape[1]*shape[2]*shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, W3) + B3)
        dr = tf.nn.dropout(hidden, 0.5)
        hidden = tf.nn.relu(tf.matmul(dr, W4) + B4)
        return tf.matmul(hidden, W5) + B5

    # Training Computation
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)
    )

    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # predictions for the training, validation and test data
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction  = tf.nn.softmax(model(tf_test_dataset))

num_steps = 20001

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


'''
/Users/hobart/envs/learn/bin/python3.6 /Users/hobart/src/python/tfstudy/udacity/5_assignment4_CNN_best_try.py
Training set (274491, 28, 28) (274491,)
Validation set (15249, 28, 28) (15249,)
Test set (14750, 28, 28) (14750,)
Training set (274491, 28, 28, 1) (274491, 10)
Validation set (15249, 28, 28, 1) (15249, 10)
Test set (14750, 28, 28, 1) (14750, 10)
2017-09-25 21:33:33.836871: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-09-25 21:33:33.836891: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
Initialized
Minibatch loss at step 0: 4.388234
Minibatch accuracy: 13.3%
Validation accuracy: 10.4%
Minibatch loss at step 500: 0.796540
Minibatch accuracy: 73.4%
Validation accuracy: 77.1%
Minibatch loss at step 1000: 0.693535
Minibatch accuracy: 76.6%
Validation accuracy: 80.1%
Minibatch loss at step 1500: 0.512812
Minibatch accuracy: 82.8%
Validation accuracy: 81.4%
Minibatch loss at step 2000: 0.677284
Minibatch accuracy: 78.1%
Validation accuracy: 82.5%
Minibatch loss at step 2500: 0.431123
Minibatch accuracy: 86.7%
Validation accuracy: 82.8%
Minibatch loss at step 3000: 0.485071
Minibatch accuracy: 88.3%
Validation accuracy: 83.3%
Minibatch loss at step 3500: 0.528564
Minibatch accuracy: 83.6%
Validation accuracy: 83.9%
Minibatch loss at step 4000: 0.371533
Minibatch accuracy: 89.8%
Validation accuracy: 84.6%
Minibatch loss at step 4500: 0.339563
Minibatch accuracy: 90.6%
Validation accuracy: 84.8%
Minibatch loss at step 5000: 0.364585
Minibatch accuracy: 87.5%
Validation accuracy: 84.9%
Minibatch loss at step 5500: 0.321016
Minibatch accuracy: 90.6%
Validation accuracy: 85.2%
Minibatch loss at step 6000: 0.452502
Minibatch accuracy: 85.9%
Validation accuracy: 85.0%
Minibatch loss at step 6500: 0.301225
Minibatch accuracy: 89.1%
Validation accuracy: 85.5%
Minibatch loss at step 7000: 0.417742
Minibatch accuracy: 86.7%
Validation accuracy: 85.8%
Minibatch loss at step 7500: 0.363910
Minibatch accuracy: 88.3%
Validation accuracy: 85.6%
Minibatch loss at step 8000: 0.378018
Minibatch accuracy: 86.7%
Validation accuracy: 86.0%
Minibatch loss at step 8500: 0.416373
Minibatch accuracy: 88.3%
Validation accuracy: 86.1%
Minibatch loss at step 9000: 0.576662
Minibatch accuracy: 82.0%
Validation accuracy: 86.1%
Minibatch loss at step 9500: 0.390249
Minibatch accuracy: 90.6%
Validation accuracy: 86.5%
Minibatch loss at step 10000: 0.375915
Minibatch accuracy: 87.5%
Validation accuracy: 86.2%
Minibatch loss at step 10500: 0.374953
Minibatch accuracy: 92.2%
Validation accuracy: 86.5%
Minibatch loss at step 11000: 0.413903
Minibatch accuracy: 86.7%
Validation accuracy: 86.5%
Minibatch loss at step 11500: 0.375574
Minibatch accuracy: 86.7%
Validation accuracy: 86.4%
Minibatch loss at step 12000: 0.532574
Minibatch accuracy: 85.2%
Validation accuracy: 86.7%
Minibatch loss at step 12500: 0.314095
Minibatch accuracy: 91.4%
Validation accuracy: 86.5%
Minibatch loss at step 13000: 0.455676
Minibatch accuracy: 86.7%
Validation accuracy: 86.6%
Minibatch loss at step 13500: 0.251155
Minibatch accuracy: 92.2%
Validation accuracy: 87.2%
Minibatch loss at step 14000: 0.364939
Minibatch accuracy: 87.5%
Validation accuracy: 87.3%
Minibatch loss at step 14500: 0.256884
Minibatch accuracy: 91.4%
Validation accuracy: 87.1%
Minibatch loss at step 15000: 0.478739
Minibatch accuracy: 84.4%
Validation accuracy: 87.0%
Minibatch loss at step 15500: 0.415569
Minibatch accuracy: 87.5%
Validation accuracy: 87.2%
Minibatch loss at step 16000: 0.575020
Minibatch accuracy: 83.6%
Validation accuracy: 87.4%
Minibatch loss at step 16500: 0.579728
Minibatch accuracy: 78.9%
Validation accuracy: 87.2%
Minibatch loss at step 17000: 0.439972
Minibatch accuracy: 85.9%
Validation accuracy: 87.2%
Minibatch loss at step 17500: 0.347196
Minibatch accuracy: 90.6%
Validation accuracy: 87.4%
Minibatch loss at step 18000: 0.422618
Minibatch accuracy: 89.1%
Validation accuracy: 87.4%
Minibatch loss at step 18500: 0.358268
Minibatch accuracy: 87.5%
Validation accuracy: 87.4%
Minibatch loss at step 19000: 0.392853
Minibatch accuracy: 88.3%
Validation accuracy: 87.4%
Minibatch loss at step 19500: 0.399897
Minibatch accuracy: 89.1%
Validation accuracy: 87.7%
Minibatch loss at step 20000: 0.188165
Minibatch accuracy: 95.3%
Validation accuracy: 87.5%
Test accuracy: 93.5%
'''








