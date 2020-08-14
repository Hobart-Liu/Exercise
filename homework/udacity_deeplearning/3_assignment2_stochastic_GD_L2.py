import numpy as np
import tensorflow as tf
import pickle
from os.path import join, expanduser

mnist_root      = join(expanduser("~"), 'mldata')
data_root       = join(mnist_root, 'notmnist')
pickle_file     = join(data_root, "notMNIST_unique.pickle")

img_size = 28
num_labels = 10


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
    dataset = dataset.reshape((-1, img_size*img_size)).astype(np.float32)
    labels = (np.arange(num_labels) == labels.reshape(-1, 1)).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

batch_size = 128

graph = tf.Graph()

with graph.as_default():

    # Input data. For th traiing data, we use placeholder that will be fed
    # at run time with a training minibatch.

    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape =(batch_size, img_size*img_size))
    tf_train_labels = tf.placeholder(tf.float32,
                                     shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    lamda = tf.constant(1e-3)

    # Variables
    weights = tf.Variable(
        tf.truncated_normal([img_size * img_size, num_labels])
    )
    biases = tf.Variable(tf.zeros([num_labels]))

    # Training computations
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels, logits=logits) +
        lamda * tf.nn.l2_loss(weights)
    )

    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for training, validation, and test data
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(
        tf.matmul(tf_valid_dataset, weights) + biases
    )
    test_prediction  = tf.nn.softmax(
        tf.matmul(tf_test_dataset, weights) + biases
    )

num_steps = 3001

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")

    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch
        batch_data = train_dataset[offset:(offset+batch_size), :]
        batch_labels = train_labels[offset:(offset+batch_size), :]
        # prepare a dictionary telling the session where to feed the minibatch.
        # The key of this dictionary is the placeholder node of the graph to be fed.
        # and the value is numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data,
                     tf_train_labels : batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict
        )
        if (step % 500 == 0):
          print("Minibatch loss at step %d: %f" % (step, l))
          print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
          print("Validation accuracy: %.1f%%" % accuracy(
            valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


'''
running result

/Users/hobart/envs/learn/bin/python3.6 /Users/hobart/src/python/tfstudy/udacity/3_assignment2_stochastic_GD.py
Training set (274491, 28, 28) (274491,)
Validation set (15249, 28, 28) (15249,)
Test set (14750, 28, 28) (14750,)
Training set (274491, 784) (274491, 10)
Validation set (15249, 784) (15249, 10)
Test set (14750, 784) (14750, 10)
2017-09-23 15:02:28.372177: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-09-23 15:02:28.372200: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
Initialized
Minibatch loss at step 0: 17.225166
Minibatch accuracy: 11.7%
Validation accuracy: 13.9%
Minibatch loss at step 500: 2.114276
Minibatch accuracy: 68.8%
Validation accuracy: 72.9%
Minibatch loss at step 1000: 1.398767
Minibatch accuracy: 72.7%
Validation accuracy: 74.3%
Minibatch loss at step 1500: 1.000032
Minibatch accuracy: 80.5%
Validation accuracy: 75.1%
Minibatch loss at step 2000: 1.114583
Minibatch accuracy: 75.8%
Validation accuracy: 75.3%
Minibatch loss at step 2500: 0.866951
Minibatch accuracy: 80.5%
Validation accuracy: 75.8%
Minibatch loss at step 3000: 0.837457
Minibatch accuracy: 81.2%
Validation accuracy: 76.7%
Test accuracy: 83.7%

'''