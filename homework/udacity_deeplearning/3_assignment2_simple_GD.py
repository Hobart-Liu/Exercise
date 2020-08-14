import numpy as np
import tensorflow as tf
import pickle
from os.path import join, expanduser

mnist_root      = join(expanduser("~"), 'mldata')
data_root       = join(mnist_root, 'notmnist')
pickle_file     = join(data_root, "notMNIST_unique.pickle")

img_size = 28
num_labels = 10
num_steps = 801

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

'''
TensorFlow works like this:

    First you describe the computation that you want to see performed: 
    what the inputs, the variables, and the operations look like. T
    hese get created as nodes over a computation graph. 
    This description is all contained within the block below:

    with graph.as_default():
        ...

    Then you can run the operations on this graph as many times as you want 
    by calling session.run(), providing it outputs to fetch from the graph that get returned. 
    This runtime operation is all contained in the block below:

    with tf.Session(graph=graph) as session:
        ...


'''


train_subset = 10000

graph = tf.Graph()
with graph.as_default():

    # Input data.
    # Load the training validation and test data into constants that are
    # attached to the graph

    tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
    tf_train_labels = tf.constant(train_labels[:train_subset])
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables
    # These are the parameters that we are going to be training. The weight
    # matrix will be initialized using random values following a (truncated)
    # normal distribution. The biases get initialized to zero

    # 不知道为什么TENSORFLOW用LIST来表现SHAPE,这点与PYTHON的常规不符合，
    # 通常在NUMPY和其他的PYTHON LIBRARY里我们用（）来表达SHAPE
    # 在TENSORFLOW 1.3中大概试了一下，TENSORFLOW这两种方式现在都支持。


    weights = tf.Variable(
        tf.truncated_normal([img_size * img_size, num_labels])
    )
    biases = tf.Variable(tf.zeros([num_labels]))


    # Training computation.
    # We multiply the inputs with the weight matrix, and add biases. We compute
    # the softmax and cross-entropy (it's one operation in Tensorflow, because
    # it's very common, and it can be optimized). We take the average of this
    # cross-entropy across all training examples: that's our loss.

    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)
    )

    # Optimizer.
    # We are going to find the minimum of this loss using gradient descent
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # predictions for the training, validation and test data
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(
        tf.matmul(tf_valid_dataset, weights) + biases
    )
    test_prediction  = tf.nn.softmax(
        tf.matmul(tf_test_dataset, weights) + biases
    )

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


with tf.Session(graph=graph) as session:
    # This is a one-time operation which ensures the parameters get initialized as
    # we described in the graph: random weights for the matrix, zeros for th ebiases

    tf.global_variables_initializer().run()
    print("initialized")

    for step in range(num_steps):
        # Run the computations. We tell .run() that we want to run the optimizer
        # and get the loss value and the training predictions returned as numpy arrays.

        _, l, predictions = session.run([optimizer, loss, train_prediction])

        if (step % 100 == 0):
            print('Loss at step %d: %f' % (step, l))
            print('Training accuracy: %.1f%%' % accuracy(
                predictions, train_labels[:train_subset, :]
            ))

            # calling. .eval() on valid_porediction is basically like calling run(), but
            # just to get that one numpy array. Note that it recompute all its graph dependecies.
            print('Validation accuracy: %.1f%%' % accuracy(
                valid_prediction.eval(), valid_labels
            ))
    print('Test accuracy: %.1f%%' %accuracy(
        test_prediction.eval(), test_labels
    ))



'''
running result

Training set (274491, 28, 28) (274491,)
Validation set (15249, 28, 28) (15249,)
Test set (14750, 28, 28) (14750,)
Training set (274491, 784) (274491, 10)
Validation set (15249, 784) (15249, 10)
Test set (14750, 784) (14750, 10)
2017-09-23 15:03:12.684062: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-09-23 15:03:12.684085: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
initialized
Loss at step 0: 16.843006
Training accuracy: 9.9%
Validation accuracy: 11.8%
Loss at step 100: 2.378927
Training accuracy: 71.6%
Validation accuracy: 68.2%
Loss at step 200: 1.936043
Training accuracy: 74.5%
Validation accuracy: 70.3%
Loss at step 300: 1.684302
Training accuracy: 75.5%
Validation accuracy: 71.3%
Loss at step 400: 1.510754
Training accuracy: 76.4%
Validation accuracy: 71.9%
Loss at step 500: 1.381044
Training accuracy: 77.0%
Validation accuracy: 72.3%
Loss at step 600: 1.279034
Training accuracy: 77.7%
Validation accuracy: 72.5%
Loss at step 700: 1.196131
Training accuracy: 78.1%
Validation accuracy: 72.8%
Loss at step 800: 1.126975
Training accuracy: 78.6%
Validation accuracy: 72.8%
Test accuracy: 80.9%

'''