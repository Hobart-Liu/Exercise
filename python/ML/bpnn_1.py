import numpy as np
import math

'''

Reference:

UFLDL Tutorial: 
http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm

'''

# input and target
x = [[0,0],
     [0,1],
     [1,0],
     [1,1]]   # row * n_features
y = [[0,1],
     [1,0],
     [1,0],
     [0,0]]   # row * n_targets

def pr(name, x):
    print("==", name, "==\n")
    print(x, "\n")

# activation function

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return x * (1.0 - x)


# Network configuration

m, n_input_layer   = np.shape(x)
_, n_output_layer  = np.shape(y)

n_hidden_layer1 = 5



alpha = 0.5 # learning rate
epsilon = 0.1
w1 = np.array(np.random.normal(0,epsilon**2,size=(n_hidden_layer1, n_input_layer)), dtype=np.float64)
b1 = np.array(np.random.normal(0,epsilon**2,size=(n_hidden_layer1,1)), dtype=np.float64)
w2 = np.array(np.random.normal(0,epsilon**2,size=(n_output_layer, n_hidden_layer1)), dtype=np.float64)
b2 = np.array(np.random.normal(0,epsilon**2,size=(n_output_layer,1)), dtype= np.float64)

# Training

for iter in range(10000):

    delta_w2 = np.array(np.zeros(w2.shape))
    delta_w1 = np.array(np.zeros(w1.shape))
    delta_b2 = np.array(np.zeros(b2.shape))
    delta_b1 = np.array(np.zeros(b1.shape))
    cost = 0.0

    for item in range(m):


        # 1) Perform a feedforward pass,
        #    computing the activations for layers L2, L3,
        #    and so on up to the output layer Lm
        a1 = np.array([x[item]]).T
        z2 = np.dot(w1, a1) + b1
        a2 = sigmoid(z2)
        z3 = np.dot(w2, a2)+ b2
        a3 = sigmoid(z3)

        y_ = np.array([y[item]]).T

        cost = sum(np.power((y_-a3),2)/2)

        # 2_1)
        # For each output unit i in layer nl (the output layer), set δ
        delta_3 = np.multiply(-(y_-a3), sigmoid_prime(a3))
        # For l = n_l-1, n_l-2, n_l-3, ... , 2, set δ
        delta_2 = np.multiply(np.dot(w2.T, delta_3), sigmoid_prime(a2))

        # 2_2)
        # Compute the desired partial derivatives for W and B
        delta_w2 += np.dot(delta_3, a2.T)
        delta_w1 += np.dot(delta_2, a1.T)
        delta_b2 += delta_3
        delta_b1 += delta_2

        # 2_3)
        # Update W and B
        #w2 = w2 - alpha*(delta_w2/m + lamda*w2)
        w2 -= alpha * delta_w2
        #w1 = w1 - alpha*(delta_w1/m + lamda*w1)
        w1 -= alpha * delta_w1
        b2 -= alpha * delta_b2
        b1 -= alpha * delta_b1



    if iter % 100 == 0:
        print("cost function = ", cost)
        #pr("delta_w2", delta_w2)
        #pr("delta_w1", delta_w1)
        #pr("w2", w2)
        #pr("w1", w1)
        #pr("b2", b2)
        #pr("b1", b1)



# predict

for i in range(4):
    a1 = np.array([x[i]]).T
    z2 = np.dot(w1, a1) + b1
    a2 = sigmoid(z2)
    z3 = np.dot(w2, a2)+ b2
    a3 = sigmoid(z3)
    print(a1, a3)