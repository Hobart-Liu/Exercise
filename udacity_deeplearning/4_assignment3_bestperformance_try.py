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
num_steps = 4001
input_dim = img_size*img_size
hidden1_dim = 1024
hidden2_dim = 300
hidden3_dim = 100
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

    # Input Variables
    input_layer = tf.placeholder(tf.float32, shape=([batch_size, input_dim]))
    output_layer = tf.placeholder(tf.float32, shape=([batch_size, output_dim]))
    dr = tf.constant(0.7)
    lamda = tf.constant(1e-3)
    tf_test_dataset = tf.constant(test_dataset)
    tf_valid_dataset = tf.constant(valid_dataset)
    global_step = tf.Variable(0)  # count the number of steps taken.

    # Variable
    # TODO:如果去掉stddev,loss 值会不断增加，最后产生nan
    # The extra layer made the gradients too unstable, and that lead to the loss function quickly devolving to NaN.
    # The best way to fix this is to use Xavier initialization. Otherwise, the variance of the initial values will
    # tend to be too high, causing instability. Also, decreasing the learning rate may help.
    # try lowering the standard deviation of the weights. The default is set to 1. It worked for me
    # Yes. My learning rate was at a fixed 0.5 and I had to lower it to 0.06 to get it going.
    W1 = tf.Variable(tf.truncated_normal(shape=[input_dim, hidden1_dim], stddev=np.sqrt(2.0 / input_dim)))
    W2 = tf.Variable(tf.truncated_normal(shape=[hidden1_dim, hidden2_dim], stddev=np.sqrt(2.0 / input_dim)))
    W3 = tf.Variable(tf.truncated_normal(shape=[hidden2_dim, hidden3_dim], stddev=np.sqrt(2.0 / input_dim)))
    W4 = tf.Variable(tf.truncated_normal(shape=[hidden3_dim, output_dim], stddev=np.sqrt(2.0 / input_dim)))
    B1 = tf.zeros(shape=[hidden1_dim])
    B2 = tf.zeros(shape=[hidden2_dim])
    B3 = tf.zeros(shape=[hidden3_dim])
    B4 = tf.zeros(shape=[output_dim])

    # Training computation
    hidden1_layer = tf.nn.relu(tf.matmul(input_layer, W1) + B1)
    drop1         = tf.nn.dropout(hidden1_layer, dr)
    hidden2_layer = tf.nn.relu(tf.matmul(drop1, W2) + B2)
    drop2         = tf.nn.dropout(hidden2_layer, dr)
    hidden3_layer = tf.nn.relu(tf.matmul(drop2, W3) + B3)
    drop3         = tf.nn.dropout(hidden3_layer, dr)
    logit         = tf.matmul(drop3, W4) + B4
    learning_rate = tf.train.exponential_decay(0.3, global_step, 100, 0.9, staircase=True)

    # loss function
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels = output_layer, logits=logit) + \
        lamda * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3))
    )

    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    #predictions
    pred_train = tf.nn.softmax(logit)

    lay1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, W1) + B1)
    lay2_valid = tf.nn.relu(tf.matmul(lay1_valid, W2) + B2)
    lay3_valid = tf.nn.relu(tf.matmul(lay2_valid, W3) + B3)
    pred_valid = tf.nn.softmax(tf.matmul(lay3_valid, W4) + B4)

    lay1_test = tf.nn.relu(tf.matmul(tf_test_dataset, W1) + B1)
    lay2_test = tf.nn.relu(tf.matmul(lay1_test, W2) + B2)
    lay3_test = tf.nn.relu(tf.matmul(lay2_test, W3) + B3)
    pred_test = tf.nn.softmax(tf.matmul(lay3_test, W4) + B4)

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")

    for step in range(num_steps):

        offset = (step * batch_size) % (test_dataset.shape[0] - batch_size)

        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]

        feed_dict = {input_layer: batch_data,
                     output_layer: batch_labels}

        _, l, predictions = session.run([optimizer, loss, pred_train], feed_dict=feed_dict)

        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(pred_valid.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(pred_test.eval(), test_labels))



    # Seeing is Believing

    pred = pred_test.eval()

    pretty_labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J'}
    items = random.sample(range(len(pred)), 8)
    for i, item in enumerate(items):
        plt.subplot(2, 4, i + 1)
        plt.axis('off')
        plt.title(pretty_labels[pred[item].argmax()])
        plt.imshow(test_dataset[item].reshape(img_size, img_size))

    plt.show()

