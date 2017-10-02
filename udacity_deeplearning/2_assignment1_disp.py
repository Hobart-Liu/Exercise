import pickle
import numpy as np
from os.path import join, expanduser


mnist_root      = join(expanduser("~"), 'mldata')
data_root       = join(mnist_root, 'notmnist')
pickle_file     = join(data_root, "notMNIST.pickle")
verify_file     = join(data_root, "verify.png")

img_size = 28
pixel_depth = 255.0

with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

train_dataset = data['train_dataset']
train_label = data['train_labels']
valid_dataset = data['valid_dataset']
valid_label = data['valid_labels']
test_dataset = data['test_dataset']
test_label = data['test_labels']

del data
print(train_dataset.shape, train_label.shape)
print(valid_dataset.shape, valid_label.shape)
print(test_dataset.shape, test_label.shape)


# 方法用于作图
from udacity.util import show_images
displist = []
l1, l2, l3 = len(train_label), len(valid_label), len(test_label)
x1, x2, x3 = np.random.choice(l1, 10), np.random.choice(l2, 10), np.random.choice(l3, 10)
displist.append(train_dataset[x1])
displist.append(valid_dataset[x2])
displist.append(test_dataset[x3])

show_images(displist, 3, 10, verify_file, True)

