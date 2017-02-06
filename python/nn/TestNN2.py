import pandas as pd
import numpy as np
from NN import *


store = pd.HDFStore('mnist.h5', 'r')
df = store.get('train')
store.close()
test = np.matrix(df[0:9999])
valid = np.matrix(df[10000:12999])
train = np.matrix(df[13000:])

n_test_records  = len(test)
n_valid_records = len(valid)
n_train_records = len(train)

n_shape = 28*28
n_label = 10


def get_next_batch(data, num = 100, batch=0):
    # pick lable information
    tmp = data[batch*num:(batch+1)*num, 0]
    lbl = np.matrix(np.zeros([num,n_label]))
    for r in range(num):
        lbl[r,tmp[r,0]]=1
    # pick image information
    img  = data[batch*num:(batch+1)*num, 1:]
    #increase batch
    if ((batch+2) * num - 1) > n_train_records:
        batch = 0
    else:
        batch += 1

    '''
    Note: lbl (one hot) and img format

    ----y1----    ----x1----
    ----y2----    ----x2----
    ----y3----    ----x3----
    [num * 10]    [num * 784]

    '''
    return lbl, img, batch

def txt(img, threshold=200):
    render = ''
    w, h = img.shape
    for i in range(w):
        for j in range(h):
            if img[i,j] > threshold:
                render += '@@'
            else:
                render += '  '
        render += '\n'
    return render


bpnn1 = neural_network_with_one_hidden_layer(n_shape,40,n_label,learning_rate = 0.1)

valid_lbl, valid_img, _ = get_next_batch(valid, num=n_valid_records, batch=0)
valid_x = valid_img.T
valid_y = valid_lbl.T

batch = 0
for i in range(50000):
    lbl, img, batch = get_next_batch(train, num=100, batch=batch)
    tr_y = lbl.T
    tr_x = img.T

    train_error = bpnn1.train_one_iteration(tr_x, tr_y)
    bpnn1.reset_deltas()
    if i % 1000 == 0:
        valid_error, rate = bpnn1.test(valid_x, valid_y)
        print("training error", train_error, "\t\t validation error", valid_error, "\t\t rate of correct", rate)




lbl, img, _ = get_next_batch(test, num=n_test_records, batch=0)
test_y = lbl.T
test_x = img.T

print("======== Summary =======================")
cost, rate = bpnn1.test(test_x, test_y)
print("            Cost = ", cost)
print("Correctness rate = ", rate)


# visualize the checking

for i in range(10):
    rnd = np.random.randint(0, n_test_records)
    _,_,_, output = bpnn1.feedforward(test_x[:,rnd])
    print("predict output: ", output.argmax(0), " target: ", test_y[:,rnd].argmax(0))
    print(txt(test_x[:,rnd].T.reshape(28,28)))

