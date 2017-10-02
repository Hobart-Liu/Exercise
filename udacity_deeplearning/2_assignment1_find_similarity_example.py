import pickle
from os.path import join, expanduser
import numpy as np
import hashlib
import time

# this code give sample for cosin_similarity

mnist_root      = join(expanduser("~"), 'mldata')
data_root       = join(mnist_root, 'notmnist')
u_pickle_file   = join(data_root, "unique_notMNIST.pickle")
similar_imgs    = join(data_root, "similar_images.png")

img_size = 28
pixel_depth = 255.0

with open(u_pickle_file, 'rb') as f:
    train = pickle.load(f)

# hash_train = {hashlib.sha1(x): x for x in train}

def find_similar_pairs(matrix, threshold=0.96):
    t0 = time.time()
    n = len(matrix)
    matrix = matrix.reshape(n, -1)
    matrix /= np.linalg.norm(matrix, axis=1).reshape(n, 1)

    similar_pair = []
    for i in range(n):
        for j in range(i+1, n):
            similarity = matrix[i].dot(matrix[j])
            if similarity >= threshold:
                similar_pair.append((i, j))
                print("pair", i, j, similarity)
        print("round %d" %i)


    t1 = time.time()
    print("Time: %5.2f" %(t1-t0))

    return similar_pair


print(train.shape)
testdata = train[:1000].copy()
similar_pairs = find_similar_pairs(testdata)
print(len(similar_pairs))
print(train.shape)


from udacity.util import show_images
# pics = []
# id = 0
# for i in range(10):
#     pic_row = []
#     for j in range(10):
#         id1, id2 = similar_pairs[id]
#         pic_row.append(train[id1])
#         pic_row.append(train[id2])
#         id += 1
#     pics.append(pic_row)
#
# show_images(pics, 10, 20, similar_imgs, True)


# pick 5 pairs to check the content

pics = []
id=0

for i in range(10):
    id1, id2 = similar_pairs[i]
    pic = []
    pic.append(train[id1])
    pic.append(train[id2])
    print(id1, id2)
    print(sum(sum(train[id1])), sum(sum(train[id2])))
    if (train[id1] == train[id2]).all():
        print("\n")
        print("hmmm, identical pairs")
        print(hashlib.sha1(train[id1]))
        print(hashlib.sha1(train[id2]))
        print("\n")
    pics.append(pic)

show_images(pics, 10, 2, similar_imgs, True)



