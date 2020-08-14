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


lamdas = np.array([pow(10, i) for i in np.arange(-4, -2, 0.1)]).astype(np.float32)
accuracylist = []

with tf.Session(graph=graph) as session:
    for lamda_v in lamdas:
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
                         lamda : lamda_v}
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict
            )
            if (step % 500 == 0):
              print("Minibatch loss at step %d: %f" % (step, l))
              print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
              print("Validation accuracy: %.1f%%" % accuracy(
                valid_prediction.eval(), valid_labels))
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
        accuracylist.append(accuracy(test_prediction.eval(), test_labels))

    # Seeing is Believing

plt.semilogx(lamdas, accuracylist)
plt.grid(True)
plt.title("Test accuracy by regularization")
plt.show()


'''
Running result
/Users/hobart/envs/learn/bin/python3.6 /Users/hobart/src/python/tfstudy/udacity/4_assignment3_stochastic_GD_hidden_relu_L2_multiple.py
Training set (274491, 28, 28) (274491,)
Validation set (15249, 28, 28) (15249,)
Test set (14750, 28, 28) (14750,)
Training set (274491, 784) (274491, 10)
Validation set (15249, 784) (15249, 10)
Test set (14750, 784) (14750, 10)
2017-09-23 22:04:39.873046: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-09-23 22:04:39.873068: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
Initialized
Minibatch loss at step 0: 352.522736
Minibatch accuracy: 12.5%
Validation accuracy: 35.8%
Minibatch loss at step 500: 45.370190
Minibatch accuracy: 75.8%
Validation accuracy: 78.1%
Minibatch loss at step 1000: 35.315567
Minibatch accuracy: 78.9%
Validation accuracy: 77.1%
Minibatch loss at step 1500: 33.809227
Minibatch accuracy: 83.6%
Validation accuracy: 77.3%
Minibatch loss at step 2000: 29.297718
Minibatch accuracy: 82.0%
Validation accuracy: 78.7%
Minibatch loss at step 2500: 27.220383
Minibatch accuracy: 80.5%
Validation accuracy: 81.0%
Minibatch loss at step 3000: 24.138500
Minibatch accuracy: 78.1%
Validation accuracy: 79.6%
Test accuracy: 86.6%
Initialized
Minibatch loss at step 0: 382.199738
Minibatch accuracy: 5.5%
Validation accuracy: 32.1%
Minibatch loss at step 500: 52.348976
Minibatch accuracy: 75.8%
Validation accuracy: 75.7%
Minibatch loss at step 1000: 40.722927
Minibatch accuracy: 77.3%
Validation accuracy: 78.8%
Minibatch loss at step 1500: 36.802448
Minibatch accuracy: 80.5%
Validation accuracy: 77.8%
Minibatch loss at step 2000: 33.420303
Minibatch accuracy: 80.5%
Validation accuracy: 79.1%
Minibatch loss at step 2500: 29.753345
Minibatch accuracy: 84.4%
Validation accuracy: 80.5%
Minibatch loss at step 3000: 27.635483
Minibatch accuracy: 75.0%
Validation accuracy: 79.7%
Test accuracy: 86.9%
Initialized
Minibatch loss at step 0: 427.353333
Minibatch accuracy: 6.2%
Validation accuracy: 25.4%
Minibatch loss at step 500: 56.284733
Minibatch accuracy: 74.2%
Validation accuracy: 78.2%
Minibatch loss at step 1000: 51.466316
Minibatch accuracy: 75.0%
Validation accuracy: 78.3%
Minibatch loss at step 1500: 43.220787
Minibatch accuracy: 83.6%
Validation accuracy: 77.9%
Minibatch loss at step 2000: 37.378239
Minibatch accuracy: 78.9%
Validation accuracy: 78.3%
Minibatch loss at step 2500: 33.456276
Minibatch accuracy: 84.4%
Validation accuracy: 81.1%
Minibatch loss at step 3000: 30.572008
Minibatch accuracy: 79.7%
Validation accuracy: 80.2%
Test accuracy: 87.0%
Initialized
Minibatch loss at step 0: 415.817902
Minibatch accuracy: 7.0%
Validation accuracy: 31.6%
Minibatch loss at step 500: 70.451782
Minibatch accuracy: 75.8%
Validation accuracy: 76.3%
Minibatch loss at step 1000: 54.657486
Minibatch accuracy: 78.9%
Validation accuracy: 78.9%
Minibatch loss at step 1500: 52.162014
Minibatch accuracy: 81.2%
Validation accuracy: 79.0%
Minibatch loss at step 2000: 45.209263
Minibatch accuracy: 75.0%
Validation accuracy: 77.9%
Minibatch loss at step 2500: 38.464302
Minibatch accuracy: 82.0%
Validation accuracy: 79.8%
Minibatch loss at step 3000: 34.056076
Minibatch accuracy: 80.5%
Validation accuracy: 80.8%
Test accuracy: 87.7%
Initialized
Minibatch loss at step 0: 575.326843
Minibatch accuracy: 8.6%
Validation accuracy: 38.0%
Minibatch loss at step 500: 78.664810
Minibatch accuracy: 75.8%
Validation accuracy: 77.5%
Minibatch loss at step 1000: 66.561661
Minibatch accuracy: 80.5%
Validation accuracy: 78.5%
Minibatch loss at step 1500: 57.509476
Minibatch accuracy: 79.7%
Validation accuracy: 78.4%
Minibatch loss at step 2000: 49.346107
Minibatch accuracy: 78.9%
Validation accuracy: 78.3%
Minibatch loss at step 2500: 42.175392
Minibatch accuracy: 82.0%
Validation accuracy: 81.4%
Minibatch loss at step 3000: 36.392750
Minibatch accuracy: 82.8%
Validation accuracy: 80.7%
Test accuracy: 87.3%
Initialized
Minibatch loss at step 0: 461.351868
Minibatch accuracy: 3.9%
Validation accuracy: 32.5%
Minibatch loss at step 500: 95.197487
Minibatch accuracy: 74.2%
Validation accuracy: 78.8%
Minibatch loss at step 1000: 76.236877
Minibatch accuracy: 78.9%
Validation accuracy: 78.4%
Minibatch loss at step 1500: 63.542435
Minibatch accuracy: 79.7%
Validation accuracy: 77.4%
Minibatch loss at step 2000: 53.077721
Minibatch accuracy: 78.1%
Validation accuracy: 79.1%
Minibatch loss at step 2500: 44.567497
Minibatch accuracy: 83.6%
Validation accuracy: 81.4%
Minibatch loss at step 3000: 37.463608
Minibatch accuracy: 84.4%
Validation accuracy: 81.9%
Test accuracy: 88.7%
Initialized
Minibatch loss at step 0: 459.670532
Minibatch accuracy: 16.4%
Validation accuracy: 37.2%
Minibatch loss at step 500: 111.013184
Minibatch accuracy: 77.3%
Validation accuracy: 75.1%
Minibatch loss at step 1000: 88.254150
Minibatch accuracy: 81.2%
Validation accuracy: 78.6%
Minibatch loss at step 1500: 69.295059
Minibatch accuracy: 84.4%
Validation accuracy: 78.2%
Minibatch loss at step 2000: 57.042885
Minibatch accuracy: 78.9%
Validation accuracy: 78.4%
Minibatch loss at step 2500: 44.923180
Minibatch accuracy: 85.2%
Validation accuracy: 81.2%
Minibatch loss at step 3000: 36.930729
Minibatch accuracy: 82.8%
Validation accuracy: 82.5%
Test accuracy: 89.4%
Initialized
Minibatch loss at step 0: 534.602905
Minibatch accuracy: 11.7%
Validation accuracy: 32.2%
Minibatch loss at step 500: 130.111115
Minibatch accuracy: 77.3%
Validation accuracy: 78.5%
Minibatch loss at step 1000: 98.925400
Minibatch accuracy: 77.3%
Validation accuracy: 78.8%
Minibatch loss at step 1500: 74.901947
Minibatch accuracy: 81.2%
Validation accuracy: 79.4%
Minibatch loss at step 2000: 56.983727
Minibatch accuracy: 78.1%
Validation accuracy: 80.0%
Minibatch loss at step 2500: 43.846638
Minibatch accuracy: 84.4%
Validation accuracy: 81.8%
Minibatch loss at step 3000: 34.161621
Minibatch accuracy: 84.4%
Validation accuracy: 82.1%
Test accuracy: 88.9%
Initialized
Minibatch loss at step 0: 503.899536
Minibatch accuracy: 11.7%
Validation accuracy: 40.8%
Minibatch loss at step 500: 150.093307
Minibatch accuracy: 75.8%
Validation accuracy: 77.6%
Minibatch loss at step 1000: 109.001076
Minibatch accuracy: 75.8%
Validation accuracy: 79.7%
Minibatch loss at step 1500: 76.569763
Minibatch accuracy: 80.5%
Validation accuracy: 78.8%
Minibatch loss at step 2000: 54.924950
Minibatch accuracy: 81.2%
Validation accuracy: 79.7%
Minibatch loss at step 2500: 39.392029
Minibatch accuracy: 86.7%
Validation accuracy: 82.3%
Minibatch loss at step 3000: 28.837418
Minibatch accuracy: 86.7%
Validation accuracy: 83.4%
Test accuracy: 90.4%
Initialized
Minibatch loss at step 0: 667.376343
Minibatch accuracy: 5.5%
Validation accuracy: 33.3%
Minibatch loss at step 500: 171.514404
Minibatch accuracy: 77.3%
Validation accuracy: 77.3%
Minibatch loss at step 1000: 113.753738
Minibatch accuracy: 84.4%
Validation accuracy: 78.6%
Minibatch loss at step 1500: 75.082191
Minibatch accuracy: 81.2%
Validation accuracy: 78.8%
Minibatch loss at step 2000: 50.161804
Minibatch accuracy: 80.5%
Validation accuracy: 81.4%
Minibatch loss at step 2500: 33.350666
Minibatch accuracy: 84.4%
Validation accuracy: 83.6%
Minibatch loss at step 3000: 22.450367
Minibatch accuracy: 86.7%
Validation accuracy: 84.5%
Test accuracy: 91.4%
Initialized
Minibatch loss at step 0: 663.386536
Minibatch accuracy: 10.9%
Validation accuracy: 33.5%
Minibatch loss at step 500: 196.897125
Minibatch accuracy: 76.6%
Validation accuracy: 76.4%
Minibatch loss at step 1000: 115.936890
Minibatch accuracy: 75.0%
Validation accuracy: 79.5%
Minibatch loss at step 1500: 68.888474
Minibatch accuracy: 81.2%
Validation accuracy: 79.8%
Minibatch loss at step 2000: 41.548336
Minibatch accuracy: 82.0%
Validation accuracy: 83.1%
Minibatch loss at step 2500: 25.189184
Minibatch accuracy: 86.7%
Validation accuracy: 84.5%
Minibatch loss at step 3000: 15.362730
Minibatch accuracy: 87.5%
Validation accuracy: 85.4%
Test accuracy: 92.0%
Initialized
Minibatch loss at step 0: 739.164734
Minibatch accuracy: 10.9%
Validation accuracy: 32.8%
Minibatch loss at step 500: 214.494629
Minibatch accuracy: 78.1%
Validation accuracy: 78.5%
Minibatch loss at step 1000: 112.333405
Minibatch accuracy: 76.6%
Validation accuracy: 79.5%
Minibatch loss at step 1500: 58.630329
Minibatch accuracy: 85.9%
Validation accuracy: 81.0%
Minibatch loss at step 2000: 31.565102
Minibatch accuracy: 78.9%
Validation accuracy: 83.6%
Minibatch loss at step 2500: 16.777260
Minibatch accuracy: 89.1%
Validation accuracy: 85.3%
Minibatch loss at step 3000: 9.232709
Minibatch accuracy: 86.7%
Validation accuracy: 86.0%
Test accuracy: 92.3%
Initialized
Minibatch loss at step 0: 916.441467
Minibatch accuracy: 14.8%
Validation accuracy: 30.3%
Minibatch loss at step 500: 229.371490
Minibatch accuracy: 71.9%
Validation accuracy: 74.5%
Minibatch loss at step 1000: 100.750198
Minibatch accuracy: 75.8%
Validation accuracy: 80.9%
Minibatch loss at step 1500: 45.120052
Minibatch accuracy: 85.9%
Validation accuracy: 82.2%
Minibatch loss at step 2000: 20.741407
Minibatch accuracy: 81.2%
Validation accuracy: 84.0%
Minibatch loss at step 2500: 9.586969
Minibatch accuracy: 88.3%
Validation accuracy: 85.1%
Minibatch loss at step 3000: 4.688544
Minibatch accuracy: 86.7%
Validation accuracy: 85.8%
Test accuracy: 92.1%
Initialized
Minibatch loss at step 0: 894.860229
Minibatch accuracy: 16.4%
Validation accuracy: 32.7%
Minibatch loss at step 500: 233.104614
Minibatch accuracy: 73.4%
Validation accuracy: 77.5%
Minibatch loss at step 1000: 84.087158
Minibatch accuracy: 78.1%
Validation accuracy: 81.4%
Minibatch loss at step 1500: 30.849495
Minibatch accuracy: 89.1%
Validation accuracy: 83.5%
Minibatch loss at step 2000: 11.859803
Minibatch accuracy: 81.2%
Validation accuracy: 84.8%
Minibatch loss at step 2500: 4.649326
Minibatch accuracy: 86.7%
Validation accuracy: 85.4%
Minibatch loss at step 3000: 2.083130
Minibatch accuracy: 88.3%
Validation accuracy: 85.6%
Test accuracy: 92.0%
Initialized
Minibatch loss at step 0: 1181.221436
Minibatch accuracy: 7.8%
Validation accuracy: 34.9%
Minibatch loss at step 500: 224.456512
Minibatch accuracy: 68.8%
Validation accuracy: 77.4%
Minibatch loss at step 1000: 62.949554
Minibatch accuracy: 78.9%
Validation accuracy: 82.4%
Minibatch loss at step 1500: 18.188414
Minibatch accuracy: 86.7%
Validation accuracy: 84.1%
Minibatch loss at step 2000: 5.758910
Minibatch accuracy: 79.7%
Validation accuracy: 84.2%
Minibatch loss at step 2500: 1.972820
Minibatch accuracy: 89.8%
Validation accuracy: 84.9%
Minibatch loss at step 3000: 0.982613
Minibatch accuracy: 88.3%
Validation accuracy: 85.4%
Test accuracy: 91.8%
Initialized
Minibatch loss at step 0: 1255.504150
Minibatch accuracy: 13.3%
Validation accuracy: 32.6%
Minibatch loss at step 500: 203.884552
Minibatch accuracy: 78.1%
Validation accuracy: 77.3%
Minibatch loss at step 1000: 41.721443
Minibatch accuracy: 80.5%
Validation accuracy: 83.3%
Minibatch loss at step 1500: 8.993979
Minibatch accuracy: 85.2%
Validation accuracy: 84.4%
Minibatch loss at step 2000: 2.492082
Minibatch accuracy: 78.9%
Validation accuracy: 83.8%
Minibatch loss at step 2500: 0.934725
Minibatch accuracy: 86.7%
Validation accuracy: 84.5%
Minibatch loss at step 3000: 0.659286
Minibatch accuracy: 88.3%
Validation accuracy: 84.9%
Test accuracy: 91.2%
Initialized
Minibatch loss at step 0: 1537.162109
Minibatch accuracy: 12.5%
Validation accuracy: 32.4%
Minibatch loss at step 500: 168.549042
Minibatch accuracy: 80.5%
Validation accuracy: 79.9%
Minibatch loss at step 1000: 23.432348
Minibatch accuracy: 82.0%
Validation accuracy: 84.3%
Minibatch loss at step 1500: 3.720358
Minibatch accuracy: 84.4%
Validation accuracy: 83.7%
Minibatch loss at step 2000: 1.220284
Minibatch accuracy: 78.1%
Validation accuracy: 83.4%
Minibatch loss at step 2500: 0.651077
Minibatch accuracy: 89.8%
Validation accuracy: 84.2%
Minibatch loss at step 3000: 0.591902
Minibatch accuracy: 88.3%
Validation accuracy: 84.6%
Test accuracy: 90.9%
Initialized
Minibatch loss at step 0: 1914.838623
Minibatch accuracy: 11.7%
Validation accuracy: 28.1%
Minibatch loss at step 500: 126.666374
Minibatch accuracy: 82.0%
Validation accuracy: 80.8%
Minibatch loss at step 1000: 10.930984
Minibatch accuracy: 83.6%
Validation accuracy: 84.0%
Minibatch loss at step 1500: 1.463748
Minibatch accuracy: 85.2%
Validation accuracy: 83.0%
Minibatch loss at step 2000: 0.858096
Minibatch accuracy: 77.3%
Validation accuracy: 83.0%
Minibatch loss at step 2500: 0.612238
Minibatch accuracy: 85.9%
Validation accuracy: 83.8%
Minibatch loss at step 3000: 0.606554
Minibatch accuracy: 85.9%
Validation accuracy: 84.0%
Test accuracy: 90.2%
Initialized
Minibatch loss at step 0: 2493.249512
Minibatch accuracy: 10.9%
Validation accuracy: 36.7%
Minibatch loss at step 500: 83.255127
Minibatch accuracy: 82.0%
Validation accuracy: 81.9%
Minibatch loss at step 1000: 4.293463
Minibatch accuracy: 82.0%
Validation accuracy: 83.1%
Minibatch loss at step 1500: 0.805134
Minibatch accuracy: 82.0%
Validation accuracy: 82.5%
Minibatch loss at step 2000: 0.814343
Minibatch accuracy: 78.1%
Validation accuracy: 82.4%
Minibatch loss at step 2500: 0.637731
Minibatch accuracy: 85.2%
Validation accuracy: 83.1%
Minibatch loss at step 3000: 0.630600
Minibatch accuracy: 87.5%
Validation accuracy: 83.3%
Test accuracy: 89.5%
Initialized
Minibatch loss at step 0: 2804.470947
Minibatch accuracy: 11.7%
Validation accuracy: 30.3%
Minibatch loss at step 500: 46.464149
Minibatch accuracy: 78.9%
Validation accuracy: 83.3%
Minibatch loss at step 1000: 1.654025
Minibatch accuracy: 80.5%
Validation accuracy: 82.7%
Minibatch loss at step 1500: 0.698314
Minibatch accuracy: 82.0%
Validation accuracy: 81.9%
Minibatch loss at step 2000: 0.833198
Minibatch accuracy: 77.3%
Validation accuracy: 81.8%
Minibatch loss at step 2500: 0.676411
Minibatch accuracy: 85.2%
Validation accuracy: 82.8%
Minibatch loss at step 3000: 0.651012
Minibatch accuracy: 87.5%
Validation accuracy: 82.7%
Test accuracy: 89.1%
2017-09-23 22:45:45.663 python3.6[5073:255260] IMKClient Stall detected, *please Report* your user scenario attaching a spindump (or sysdiagnose) that captures the problem - (imkxpc_attributesForCharacterIndex:reply:) block performed very slowly (1.42 secs).

Process finished with exit code 0

'''