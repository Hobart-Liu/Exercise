from sklearn.linear_model import LogisticRegression
from os.path import join, expanduser
import pickle
import time
import matplotlib.pyplot as plt
import random

mnist_root      = join(expanduser("~"), 'mldata')
data_root       = join(mnist_root, 'notmnist')
pickle_file     = join(data_root, "notMNIST_unique.pickle")


img_size = 28
pixel_depth = 255.0
flatten_size = img_size*img_size

with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

train_dataset = data['train_dataset']
train_label = data['train_labels']
valid_dataset = data['valid_dataset']
valid_label = data['valid_labels']
test_dataset = data['test_dataset']
test_label = data['test_labels']

def random_disp_dataset(dataset, labels):
    pretty_labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J'}
    items = random.sample(range(len(labels)), 8)
    for i, item in enumerate(items):
        plt.subplot(2, 4, i + 1)
        plt.axis('off')
        plt.title(pretty_labels[labels[item]])
        plt.imshow(dataset[item])


n_samples = [50, 100, 1000, 5000,2000000]

for n_sample in n_samples:
    t1 = time.time()
    clf = LogisticRegression()
    # clf = LogisticRegression(solver='saga')
    train_x = train_dataset[:n_sample].reshape(-1, flatten_size)
    train_y = train_label[:n_sample]
    test_x = test_dataset[:n_sample].reshape(-1, flatten_size)
    test_y = test_label[:n_sample]
    clf.fit(train_x, train_y)
    t2 = time.time()
    print("Batch %4d Time %5.2f, accuracy %7.5f" % (n_sample, t2 - t1, clf.score(test_x, test_y)))
    if n_sample == n_samples[-1]:
        pred = clf.predict(test_x)
        random_disp_dataset(test_dataset, pred)
        plt.show()
    del clf

#
# Batch   50 Time  0.05, accuracy 0.68000
# Batch  100 Time  0.12, accuracy 0.73000
# Batch 1000 Time  2.64, accuracy 0.80500
# Batch 5000 Time 28.05, accuracy 0.82960
# Batch 2000000 Time 2341.80, accuracy 0.87993

