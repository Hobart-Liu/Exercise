import numpy as np


class neural_network_with_one_hidden_layer:
    def __init__(self,
                 n_input_layer,
                 n_hidden_layer,
                 n_output_layer,
                 learning_rate = 0.2,
                 epsilon = 0.01):

        self.alpha= learning_rate

        self.n_input_layer = n_input_layer
        self.n_hidden_layer = n_hidden_layer
        self.n_output_layer = n_output_layer

        self.w1 = np.matrix(np.random.normal(0, epsilon ** 2, size=(n_hidden_layer, n_input_layer)))
        self.b1 = np.matrix(np.random.normal(0, epsilon ** 2, size=(n_hidden_layer, 1)))
        self.w2 = np.matrix(np.random.normal(0, epsilon ** 2, size=(n_output_layer, n_hidden_layer)))
        self.b2 = np.matrix(np.random.normal(0, epsilon ** 2, size=(n_output_layer, 1)))

        self.delta_w2 = np.matrix(np.zeros(self.w2.shape))
        self.delta_w1 = np.matrix(np.zeros(self.w1.shape))
        self.delta_b2 = np.matrix(np.zeros(self.b2.shape))
        self.delta_b1 = np.matrix(np.zeros(self.b1.shape))

    @staticmethod
    def sigmoid(x):
        return 1.0/(1.0 + np.exp(-x))

    @staticmethod
    def sigmoid_prime(x):
        return np.multiply(x, (1.0 - x))

    '''
    each train_data and target_data
    shall in a form of vector, e.g.

           |        |
           |        |
           X        Y
           |        |
           |        |

    '''



    def feedforward(self, x):
        _, m = x.shape
        a1 = x
        z2 = self.w1 * a1 + self.b1
        a2 = self.sigmoid(z2)
        z3 = self.w2 * a2 + self.b2
        a3 = self.sigmoid(z3)
        return m, a1, a2, a3


    def reset_deltas(self):
        self.delta_w2 = np.matrix(np.zeros(self.w2.shape))
        self.delta_w1 = np.matrix(np.zeros(self.w1.shape))
        self.delta_b2 = np.matrix(np.zeros(self.b2.shape))
        self.delta_b1 = np.matrix(np.zeros(self.b1.shape))


    def test(self, test_x, test_y):
        m, _, _, output = self.feedforward(test_x)  # m is number of training records
        # here we use cross entropy
        error = np.mean(np.multiply(-np.log(output), test_y))
        error_rate = np.mean(output.argmax(0) == test_y.argmax(0))


        return error, error_rate

    def train_one_iteration(self, train_x, train_y):

        m, a1, a2, a3 = self.feedforward(train_x)  # m is number of training records

        for t in range(m):
            # 2_1)
            # For each output unit i in layer nl (the output layer), set δ
            delta_3 = -(train_y[:,t] - a3[:,t])
            # For l = n_l-1, n_l-2, n_l-3, ... , 2, set δ
            delta_2 = np.multiply(self.w2.T * delta_3, self.sigmoid_prime(a2[:,t]))

            # 2_2)
            # Compute the desired partial derivatives for W and B
            self.delta_w2 += delta_3 * a2[:,t].T
            self.delta_w1 += delta_2 * a1[:,t].T
            self.delta_b2 += delta_3
            self.delta_b1 += delta_2

        self.w2 -= self.alpha * self.delta_w2 / m
        self.w1 -= self.alpha * self.delta_w1 / m
        self.b2 -= self.alpha * self.delta_b2 / m
        self.b1 -= self.alpha * self.delta_b1 / m

        train_error = np.mean(np.multiply(-np.log(a3),train_y))

        return train_error





