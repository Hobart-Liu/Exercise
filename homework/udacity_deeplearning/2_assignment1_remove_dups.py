import pickle
from os.path import join, expanduser
import numpy as np
import hashlib

mnist_root      = join(expanduser("~"), 'mldata')
data_root       = join(mnist_root, 'notmnist')
pickle_file     = join(data_root, "notMNIST.pickle")
u_pickle_file   = join(data_root, "notMNIST_unique.pickle")
verify_file     = join(data_root, "verify.png")
dup_file        = join(data_root, "dup.png")

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


# after exploring a couple of methods I have the following notes:
# 1) Many solutions were trying hash function,
#    espeically hash.sha1() to balance the speed.
# 2) We have many ways to compute near duplicates
#    np.corrcoef calcuate corrolations between two vectors, coef = (x-mean(x))(y-mean(y))/std(x)std(y)
#    sideeffect is memory consumption. Not practical
#    image.dhash resize the picture and compute the difference between pixels
#    image.phash dot product between images
#    cosin_similarity
#    LSH local sensitive hashing.

# np.corrcoef and cosin_similarity use matrix to compute, very memory consuming
# not applicable for big data, you have to do it one by one.
# consin_similarity is common in my mind for similar vectors.

# Coding

# Remove duplicates within dataset

def get_dup_indexlist(dataset):
    temp = set()
    overlap = []
    for i, data in enumerate(dataset):
        hash = hashlib.sha1(data).hexdigest()
        if hash in temp:
            overlap.append(i)
        else:
            temp.add(hash)
    return overlap

def get_dup_indexlist2(dataset1, dataset2):
    hashset = set(hashlib.sha1(x).hexdigest() for x in dataset2)
    overlap = []
    for i, data in enumerate(dataset1):
        hash = hashlib.sha1(data).hexdigest()
        if hash in hashset:
            overlap.append(i)
    return overlap

def remove_dup(dataset1, dataset2, label1):
    if dataset2 is None:
        overlap = get_dup_indexlist(dataset1)
    else:
        overlap = get_dup_indexlist2(dataset1, dataset2)
    return np.delete(dataset1, overlap, 0), np.delete(label1, overlap, None)

train_dataset, train_label = remove_dup(train_dataset, None, train_label)
test_dataset, test_label = remove_dup(test_dataset, None, test_label)
valid_dataset, valid_label = remove_dup(valid_dataset, None, valid_label)
test_dataset, test_label = remove_dup(test_dataset, train_dataset, test_label)
valid_dataset, valid_label = remove_dup(valid_dataset, train_dataset, valid_label)
valid_dataset, valid_label = remove_dup(valid_dataset, test_dataset, valid_label)


train_hash = [hashlib.sha1(x).hexdigest() for x in train_dataset]
valid_hash = [hashlib.sha1(x).hexdigest() for x in valid_dataset]
test_hash  = [hashlib.sha1(x).hexdigest() for x in test_dataset]

train_s, valid_s, test_s = set(train_hash), set(valid_hash), set(test_hash)
print(len(train_hash), len(train_s))
print(len(valid_hash), len(valid_s))
print(len(test_hash), len(test_s))
print(train_s.intersection(valid_s))
print(valid_s.intersection(test_s))
print(test_s.intersection(train_s))

try:
    f = open(u_pickle_file, 'wb')
    save = {
        'train_dataset': train_dataset,
        'train_labels': train_label,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_label,
        'test_dataset': test_dataset,
        'test_labels': test_label
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except  Exception as e:
    print("Unable to save data to", pickle_file, ":", e)
    raise



# 一下这批代码是用来调试的
# dict 扩展 hashkey = [pic1, pic2, ...]
# Counter 用于存放 hashkey = dup_times
#
# hash_train_dict = dict()
# hash_train_dict_count = dict()
#
# for x in train_dataset:
#     key = hashlib.sha1(x).hexdigest()
#     try:
#         hash_train_dict[key]
#     except KeyError:
#         hash_train_dict[key] = []
#     hash_train_dict[key].append(x)
#
# keylist = list(hash_train_dict.keys())
#
# hash_train_dict_count = {k: len(hash_train_dict[k]) for k in keylist}
#
# # print duplicated number of each hash count
#
# from collections import Counter
#
# c = Counter(list(hash_train_dict_count.values()))
# print(c.values())
# print(c.keys())
#
# # sanity check for content with same hashcode
#
# for k, d in hash_train_dict.items():
#     if len(d) == 1:continue
#     d0 = d[0]
#     for i in range(1, len(d)):
#         if (d0 != d[i]).any(): print(k, i, len(d))
#
#
# # here I remove all duplicated picture by reassembling
# new_train_dataset = np.ndarray((len(hash_train_dict), img_size, img_size), dtype=np.float32)
# i = 0
# for k in list(hash_train_dict.keys()):
#     new_train_dataset[i, :, :] = hash_train_dict[k][0]
#     i += 1
#
# new_hash_train_dict = dict()
#
# for x in new_train_dataset:
#     key = hashlib.sha1(x).hexdigest()
#     try:
#         new_hash_train_dict[key]
#     except KeyError:
#         new_hash_train_dict[key] = []
#     new_hash_train_dict[key].append(x)
#
#
# new_keylist = list(new_hash_train_dict.keys())
#
# new_hash_train_dict_count = {k: len(new_hash_train_dict[k]) for k in new_keylist}
#
# c = Counter(list(new_hash_train_dict_count.values()))
# print(c.values())
# print(c.keys())
#
# with open(u_pickle_file, 'wb') as f:
#     pickle.dump(new_train_dataset, f, pickle.HIGHEST_PROTOCOL)
#
# exit()
#
# # randomly draw pictures
# from udacity.util import show_images
# dup_hash_list = [k for k, d in hash_train_dict_count.items() if d > 1]
#
# rows, cols = 10, 10
# dup_keys = np.random.choice(dup_hash_list, rows*cols//2)
# idx = 0
#
# pics= []
# for r in range(rows):
#     pic_row = []
#     for c in range(cols//2):
#         k = dup_keys[idx]
#         p_list = np.random.choice(range(len(hash_train_dict[k])), 2)
#         pic_row.append(hash_train_dict[k][p_list[0]])
#         pic_row.append(hash_train_dict[k][p_list[1]])
#         idx += 1
#     pics.append(pic_row)
#
# show_images(pics, rows, cols, dup_file, False)
#
#
#


# 以下方法使用np.corrcoef 判断相关性
# 体会是，不太实用，一个200k X  10k 的数据比较基本内存就崩了
# 需要不停的切片，以及组合
# 如果有其他的方法，不推荐这用方式
# ltr, lte, lva = len(train_dataset), len(test_dataset), len(valid_dataset)
#
# train_dataset = train_dataset.reshape(ltr, -1)
# test_dataset = test_dataset.reshape(lte, -1)
# valid_dataset = valid_dataset.reshape(lva, -1)
#
# print(train_dataset.shape, train_label.shape)
# print(valid_dataset.shape, valid_label.shape)
# print(test_dataset.shape, test_label.shape)
#
# x = np.vstack((train_dataset, valid_dataset))
#
# t1 = time.time()
#
# print(np.corrcoef(x[:1000]))
#
# t2 = time.time()
# print("Time: %0.2fs" %(t2-t1))


# 以下方法用于计算hash，并且通过set取得唯一值，用于判决完全一致的情况
# t1=time.time()
#
# hash_train = set([hashlib.sha1(x).hexdigest() for x in train_dataset])
# hash_valid = set([hashlib.sha1(x).hexdigest() for x in valid_dataset])
# hash_test  = set([hashlib.sha1(x).hexdigest() for x in test_dataset])
#
# t2 = time.time()
#
# overlaps_tr_va = set.intersection(hash_train, hash_valid)
# overlaps_tr_te = set.intersection(hash_train, hash_test)
# overlaps_va_te = set.intersection(hash_valid, hash_test)

########################################################################33
# after evaluate,
# I have following conclusion,
# remove identical data by using set of hash code, it is fast
# find near identical data by use cosine_similarity.

# to do the sanity check,
# I will record each picture id
# and use nested loop to do a raw data comparision
# it is slow but will ensure identical data are removed
# second, use pictures to visualize the data to double check whether they are near similarity















