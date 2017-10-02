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
batch_size = 128


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


graph = tf.Graph()

with graph.as_default():

    # Input variable

    input_layer = tf.placeholder(tf.float32, shape = (batch_size, input_dim))
    output_layer = tf.placeholder(tf.float32, shape=(batch_size, output_dim))
    dropout_rate = tf.placeholder(tf.float32)
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variable

    W1 = tf.Variable( tf.truncated_normal([input_dim, hidden1_dim]) )
    B1 = tf.Variable( tf.zeros(hidden1_dim) )
    W2 = tf.Variable( tf.truncated_normal([hidden1_dim, output_dim]) )
    B2 = tf.Variable( tf.zeros([output_dim]) )

    # Training computation

    hidden_layer = tf.nn.relu(
        tf.matmul(input_layer, W1) + B1
    )

    dropout_layer = tf.nn.dropout(hidden_layer, dropout_rate)

    logits = tf.matmul(dropout_layer, W2) + B2

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels= output_layer, logits = logits)
    )

    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # predictions of training, validation and test dataset
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(
        tf.matmul( tf.nn.relu( tf.matmul(valid_dataset, W1) + B1 ), W2) + B2
    )
    test_prediction = tf.nn.softmax(
        tf.matmul( tf.nn.relu( tf.matmul(test_dataset, W1) + B1 ), W2) + B2
    )

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

with tf.Session(graph=graph) as session:

    drop_rate_list = []
    accuracy_list = []

    for dr in np.arange(0.1, 0.9, 0.1).astype(np.float32):

        tf.global_variables_initializer().run()
        print("Initialized")

        for step in range(num_steps):

            offset = (step * batch_size) % (test_dataset.shape[0] - batch_size)

            batch_data = train_dataset[offset:(offset+batch_size), :]
            batch_labels = train_labels[offset:(offset+batch_size), :]

            feed_dict = {input_layer : batch_data,
                         output_layer : batch_labels,
                         dropout_rate : dr}

            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

            if (step % 500 == 0):
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

        drop_rate_list.append(dr)
        accuracy_list.append(accuracy(test_prediction.eval(), test_labels))




    # Seeing is Believing

    plt.plot(drop_rate_list, accuracy_list)
    plt.grid(True)
    plt.title("Test accuracy by dropout rate")
    plt.show()


"""
running result:

Initialized
Minibatch loss at step 0: 543.375610
Minibatch accuracy: 4.7%
Validation accuracy: 34.5%
Minibatch loss at step 500: 24.343681
Minibatch accuracy: 79.7%
Validation accuracy: 76.3%
Minibatch loss at step 1000: 21.287355
Minibatch accuracy: 82.0%
Validation accuracy: 78.8%
Minibatch loss at step 1500: 4.639845
Minibatch accuracy: 88.3%
Validation accuracy: 80.3%
Minibatch loss at step 2000: 9.847232
Minibatch accuracy: 84.4%
Validation accuracy: 79.7%
Minibatch loss at step 2500: 4.390558
Minibatch accuracy: 86.7%
Validation accuracy: 80.5%
Minibatch loss at step 3000: 2.490629
Minibatch accuracy: 92.2%
Validation accuracy: 79.7%
Test accuracy: 87.6%

"""