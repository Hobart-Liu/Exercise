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
num_steps = 3001
input_dim = img_size*img_size
hidden1_dim = 1024
output_dim = num_labels


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

    # input layer
    input_layer = tf.placeholder(tf.float32,
                                 shape = (batch_size, input_dim))
    output_layer = tf.placeholder(tf.float32,
                                  shape = (batch_size, output_dim))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    lamda = tf.placeholder(tf.float32)

    # Variables
    W1 = tf.Variable(
        tf.truncated_normal([input_dim, hidden1_dim])
    )
    B1 = tf.Variable(tf.zeros([hidden1_dim]))
    W2 = tf.Variable(
        tf.truncated_normal([hidden1_dim, output_dim])
    )
    B2 = tf.Variable(tf.zeros(output_dim))


    # Training computations
    hidden1 = tf.nn.relu(tf.matmul(input_layer, W1) + B1)
    logits = tf.matmul(hidden1, W2) + B2
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels = output_layer, logits=logits)+
        lamda * (
            tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)
        )
    )

    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for training, validation, and test data
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(
        tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, W1) + B1), W2) + B2
    )
    test_prediction  = tf.nn.softmax(
        tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset,  W1) + B1), W2) + B2
    )



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
        feed_dict = {input_layer : batch_data,
                     output_layer : batch_labels,
                     lamda : 1e-3}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict
        )
        if (step % 500 == 0):
          print("Minibatch loss at step %d: %f" % (step, l))
          print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
          print("Validation accuracy: %.1f%%" % accuracy(
            valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

    # Seeing is Believing

    pred = test_prediction.eval()

    pretty_labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J'}
    items = random.sample(range(len(pred)), 8)
    for i, item in enumerate(items):
        plt.subplot(2, 4, i + 1)
        plt.axis('off')
        plt.title(pretty_labels[pred[item].argmax()])
        plt.imshow(test_dataset[item].reshape(img_size, img_size))

    plt.show()


'''
Running result
Training set (274491, 28, 28) (274491,)
Validation set (15249, 28, 28) (15249,)
Test set (14750, 28, 28) (14750,)
Training set (274491, 784) (274491, 10)
Validation set (15249, 784) (15249, 10)
Test set (14750, 784) (14750, 10)
2017-09-23 20:24:33.466240: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-09-23 20:24:33.466266: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
Initialized
Minibatch loss at step 0: 717.118164
Minibatch accuracy: 10.9%
Validation accuracy: 32.3%
Minibatch loss at step 500: 194.216064
Minibatch accuracy: 76.6%
Validation accuracy: 78.1%
Minibatch loss at step 1000: 115.534988
Minibatch accuracy: 76.6%
Validation accuracy: 78.7%
Minibatch loss at step 1500: 69.200935
Minibatch accuracy: 85.2%
Validation accuracy: 80.4%
Minibatch loss at step 2000: 41.654175
Minibatch accuracy: 78.9%
Validation accuracy: 82.3%
Minibatch loss at step 2500: 25.165897
Minibatch accuracy: 86.7%
Validation accuracy: 84.6%
Minibatch loss at step 3000: 15.399674
Minibatch accuracy: 87.5%
Validation accuracy: 85.6%
Test accuracy: 91.6%


'''