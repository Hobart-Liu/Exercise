import tensorflow as tf
from os.path import join, expanduser
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import collections
import zipfile
from sklearn.manifold import TSNE

root = join(expanduser("~"), "mldata/text8")
filename = join(root, 'text8.zip')

def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
        # data = f.read(f.namelist()[0]).split()
    return data

words = read_data(filename)
print('data size is %d'  % len(words))
print(words[:20])

vocabulary_size = 50000

def build_dataset(words):

    # words 用于存储原始数据
    # dictionary 用于存放单词和排名， 把COUNT中的单词依次放入dictionary
    # reverse_dictionary, 存放排名和单词
    # data 按word中单词的出场顺序，放置排名

    # UNK token is used to denote words that are not in the dictionary
    # [count] set of tuples (word, count) with most common 50000 words
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))

    # [dictionary] 按count中单词出现顺序，为每个单词编号，建立 dict(word=idx, ...) 词典
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    #[date] 对原文进行编码，用dictionary中（Word，idx)来编码
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0 # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count  # replace -1 with unknown count

    # [reverse_dictionary] dictionary 的反向查找表
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
print('Sample data', data[:10])
print('Most common words (+UNK)', count[:5])
print(list(dictionary.items())[:5])
print(list(reverse_dictionary.items())[:5])

for i in range(10):
    print(i, words[i], reverse_dictionary[data[i]])

del words  # Hint to reduce memory.


data_index = 0

def generate_batch(batch_size, num_skips, skip_window):
    """
    function to generate a training batch for the skip-gram model.

    Example: The dog barked at the mailman.

    target: dog

    skip_window, which is the number of words back and forth from the selected word.
    if skip_window = 2, then ==> ['The', 'dog', 'barked', 'at'] will be inside the window.
    num_skips, denoting the number of different output words we will pick within the psan.
    if num_skips = 2, (skip_window=2), ('dog', 'barked'), ('dog', 'the')

    """

    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window  #skip window 要在单词左边和右边都做一遍，所以num_skips,不可以大于2*skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # e.g. if skip_window = 2, then span = 5
    # span is the length of the whole frame we are considering for a single word (left+word+right)
    # skip_window is the length of one side.
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    # queue which add and pop at the end
    buffer = collections.deque(maxlen=span)

    # get words staring from index 0 to span.
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    # 以下这段代码，我觉得写得非常乱，不是很有逻辑，至少不明显
    # 逻辑解释如下：
    # 该方法是虚拟的建立一个长度为span的列表，所有操作都以该列表的下标指针为比较的标准
    # 目标是建立batch[]包含input word, label[]包含output word
    # batch[] 的Word，始终都是buffer的中间值，skip_window正好可以用来表示这个中间值
    # （由于buffer是collections.deque,指定了固定长度，多以可以通过简单的添加动作，挤掉最前的内容）
    # 需要考虑的是label[]中的值，代码中，用target来寻找，一旦找到buffer[target]就是目标值 （dictionary的编号）
    # 我们在虚拟的长度为span的列表中开始我们的操作，该操作一共执行num_skips遍，如果不能填满batch_size,
    # 则在外部套一个循环，重复这个寻找的过程，所不同的是，target/target_to_avoid被重设了，而且buffer新塞了一个字符。
    # 首先定义target就是我们的input，即skip_window,这个值一定会被改变，所以，无所谓。
    # 我们每次都会检查是否target是否在avoidlist里，第一次一定是的，所以随机选一个randint(0, span-1),
    # 只要这个不再avoidlist里，我们成功了，添加这条训练记录，同时avoidlist里添加target记录，防止被重复挑选。


    for i in range(batch_size // num_skips):
        # 初始化的过程，targets_to_avoid 保证初始的target一定要被重选。
        target = skip_window  # target label at the center of the buffer, 这里用数组的指针来表示target
        targets_to_avoid = [skip_window] # We only need to know the words around a given word, not the word itself

        for j in range(num_skips):  # 每跳跃一次，采集一次
            # 所谓采样就是随机选了，num_skips在这里的字面意义不是很贴切， num_sampling,可能比较容易理解
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            # lucky enough, we found one
            targets_to_avoid.append(target)
            # 无语了，为每一个i分配num_skips个位置，通过j来明确定位。
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

print('data:', [reverse_dictionary[di] for di in data[:8]])

for num_skips, skip_window in [(2, 1), (4, 2)]:
    data_index = 0
    batch, labels = generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window)
    print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])
    print(batch)
    print(labels)

batch_size = 128
embedding_size = 128 # Dimension of the embedding vector.
skip_window = 1 # How many words to consider left and right.
num_skips = 2 # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16 # Random set of words to evaluate similarity on.
valid_window = 100 # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
print(valid_examples)
num_sampled = 64 # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default(), tf.device('/cpu:0'):

    # input data

    train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Variables
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)  #连续均匀分布
    )

    softmax_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0/math.sqrt(embedding_size))
    )

    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Model
    # Look up embeddings for inputs
    # embedding_loopup is used to look up corresponding embeddings of the inputs. In other words,
    # the embedding layer, is of size VxD, which will contain embedding vectors (D dimensional)
    # for all the words in the vocabulary. In order to train the model for a single instance you
    # need to find the corresponding embedding vectors for the given input words by an id lookup
    # (train_dataset in this case contain a set of unique ids corresponding to each word in the batch
    # of data). Although it possible to do this manually, use of this function is required as tensorflow
    # doesn't allow index lookup with Tensors.


    """
    这个方法的核心在于，通过编号1《-》编号2之间的强、弱关系，使得embedding[编号1】的内容得到训练
    如果不同的字词前后出现类似的词，则embedding的内容会趋于接近。（窃以为，这个和统计没太大的区别）
    
    这份代码实用价值不大，到时借此，学习了tSNE的原理。
    """

    embed = tf.nn.embedding_lookup(embeddings, train_dataset)

    # Compute the softmax loss, using a sample of the negative labels each time

    loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=embed,
                                   labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size)
    )

    # optimizer
    # Note: The optimizer will optimize the softmax_weights AND the embeddings.
    # This is because the embeddings are defined as a variable quantity and the optimizer's minimize'
    # method will by default modify all variable quantities.
    # that contributes to the tensor it is passed.
    # see docs on 'tf.train.Optimizer.minimize()' for more details.

    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

    # Compute the similarity between minibatch examples and all embeddings.
    # we use the cosine distance:
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    simiilarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

num_steps = 100001

with tf.Session(graph = graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")

    average_loss = 0

    for step in range(num_steps):
        batch_data, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {
            train_dataset: batch_data, train_labels: batch_labels
        }
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += l
        if step % 2000 == 0:
            if step > 0:
                average_loss = average_loss / 2000
            print('Average loss at step %d: %f' %(step, average_loss))
            average_loss = 0

        # note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = simiilarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log = 'Nearest to %s' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)

    final_embeddings = normalized_embeddings.eval()

num_points = 400

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :])

def plot(embeddings, labels):
  assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(15,15))  # in inches
  for i, label in enumerate(labels):
    x, y = embeddings[i,:]
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                   ha='right', va='bottom')
  plt.savefig('/Users/hobart/tmp/words.png')
  plt.show()

words = [reverse_dictionary[i] for i in range(1, num_points+1)]
plot(two_d_embeddings, words)


'''
Some Notes:
In [3]: a = collections.deque(maxlen=3)

In [4]: a
Out[4]: deque([])

In [5]: a.append(1);a
Out[5]: deque([1])

In [6]: a.append(2);a
Out[6]: deque([1, 2])

In [7]: a.append(3);a
Out[7]: deque([1, 2, 3])

In [8]: a.append(4);a
Out[8]: deque([2, 3, 4])

In [9]: a.append(5);a
Out[9]: deque([3, 4, 5])


embedding_lookup example 

In [35]: c = np.random.random([10, 1])

In [36]: b = tf.nn.embedding_lookup(c, [1, 3])

In [37]: s.run(init)

In [38]: s.run(b)
Out[38]:
array([[ 0.33930998],
       [ 0.34087243]])

In [39]: c
Out[39]:
array([[ 0.24214074],
       [ 0.33930998],
       [ 0.87790723],
       [ 0.34087243],
       [ 0.87735376],
       [ 0.22121314],
       [ 0.06933666],
       [ 0.70741758],
       [ 0.23071441],
       [ 0.93080941]])
       
reduce_mean, keep_dims
       
In [52]: c = np.arange(6.).reshape(3, 2)

In [53]: c
Out[53]:
array([[ 0.,  1.],
       [ 2.,  3.],
       [ 4.,  5.]])

In [54]: s.run(tf.reduce_mean(c, 1, keep_dims=True))
Out[54]:
array([[ 0.5],
       [ 2.5],
       [ 4.5]])


'''