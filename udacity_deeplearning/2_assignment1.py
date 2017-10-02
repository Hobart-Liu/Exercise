import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from scipy import ndimage
import glob
from sklearn.linear_model import LogisticRegression
import pickle
from udacity.util import show_image_files, show_images


num_classes = 10
np.random.seed(133)

data_file_large = os.path.join(os.path.expanduser("~"), 'mldata/notmnist/notMNIST_large.tar.gz')
data_file_small = os.path.join(os.path.expanduser("~"), 'mldata/notmnist/notMNIST_small.tar.gz')
mnist_root =os.path.join(os.path.expanduser("~"), 'mldata')
data_root = os.path.join(mnist_root, 'notmnist')
small_data_root = os.path.join(data_root, 'notMNIST_small')

img_size = 28
pixel_depth = 255.0

def maybe_extract(filename, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        print("{} already present - skipping extraction of {}".format(root, filename))
    else:
        print("Extracting data for {}".format(filename))
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(data_root)
        tar.close()

    data_folders = [os.path.join(root, d) for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % ( num_classes, len(data_folders))
        )

    print(data_folders)
    return data_folders


def load_letter(folder, min_num_images):
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), img_size, img_size), dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) - pixel_depth/2) / pixel_depth
            if image_data.shape != (img_size, img_size):
                raise Exception("Unexpected image shap: {}".format(str(image_data.shape)))
            dataset[num_images, :, :] = image_data
            num_images += 1
        except IOError as e:
            print('Could not read {}, skip'.format(image_file))

    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Fewer images than expected {} < {}'.format(num_images, min_num_images))

    print('Full dataset tensor: {}'.format(dataset.shape))
    print('Mean:', np.mean(dataset))
    print('Standard deviation', np.std(dataset))
    return dataset

def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names=[]
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            print("{} already present - Skipping pickling".format(set_filename))
        else:
            print('Pickling {}'.format(set_filename))
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print("Unbale to save data to {} due to {}".format(set_filename, e))

    return dataset_names

def show_random_image_files(data_folders):
    files = []
    for loop in range(3):
        for folder in data_folders:
            filelist = os.listdir(folder)
            filelist = np.random.choice(filelist, 50)
            new_filelist = [os.path.join(folder, f) for f in filelist]
            files.append(new_filelist)

    filename = os.path.join(data_root, 'combined.png')
    files = np.array(files)
    show_image_files(files, (28, 28), filename, True)

    return files

def show_random_images(files):
    ret = []
    n = 30
    for fi in files:
        with open(fi, 'rb') as f:
            d = pickle.load(f)
            x = list(range(len(d)))
            x1 = np.random.choice(x, n)
            ret.append(d[x1])
            x1 = np.random.choice(x, n)
            ret.append(d[x1])
            x1 = np.random.choice(x, n)
            ret.append(d[x1])

    filename = os.path.join(data_root, 'combined2.png')
    show_images(ret, len(ret), n, filename, True)


def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels

def merge_databases(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, img_size)
    train_dataset, train_labels = make_arrays(train_size, img_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class

    end_l = vsize_per_class + tsize_per_class

    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ":", e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


train_size = 300000
valid_size = 18000
test_size = 18000


test_folders = maybe_extract(data_file_small)
test_datasets = maybe_pickle(test_folders, 1800)
train_folders = maybe_extract(data_file_large)
train_datasets = maybe_pickle(train_folders, 45000)
# show_random_images(train_datasets)
valid_dataset, valid_labels, train_dataset, train_labels = merge_databases(
    train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_databases(test_datasets, test_size)

print(valid_dataset.shape, valid_labels.shape)
print(train_dataset.shape, train_labels.shape)
print(test_dataset.shape, test_labels.shape)

train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

pickle_file = os.path.join(data_root, 'notMNIST.pickle')

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except  Exception as e:
    print("Unable to save data to", pickle_file, ":", e)
    raise

statinfo = os.stat(pickle_file)
print("Compressed pickle size:", statinfo.st_size)
































