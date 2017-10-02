import tensorflow as tf
import string
import numpy as np
from os.path import join, expanduser
import os
import zipfile
import random

"""

Note 中很大一部分是借鉴了liu Sida的 学习Tensorflow的LSTM的RNN例子
https://liusida.github.io/2016/11/16/study-lstm/

"""

vocabulary_size = len(string.ascii_lowercase) + 1  # [a-z] + ' '

def check_file(filename):
    statinfo = os.stat(filename)
    assert (statinfo.st_size == 31344016)

def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        # 直接读成一个字符串，因为后面用到的就是串数据
        data = tf.compat.as_str(f.read(f.namelist()[0]))
    return data

"""
建立两个函数char2id和id2char，用来把字符对应成数字。

本程序只考虑26个字母外加1个空格字符，其他字符都当做空格来对待。
所以可以用两个函数，通过ascii码加减，直接算出对应的数值或字符。

"""

def char2id(char):
    if char in string.ascii_lowercase:
        return ord(char) - ord('a') + 1
    elif char == ' ':
        return 0
    else:
        print('Unexpected character: %s' % char)
        return 0

def id2char(dictid):
    if dictid > 0:
        return chr(dictid + ord('a') - 1)
    else:
        return ' '

"""
用 BatchGenerator.next() 方法，可以获取一批子字符串用于训练。

喂数据进神经网络的时候，先进去一批的首字符，然后再进去同一批的第二个字符，然后再进去同一批的第三个字符…

我们平时看到的向右一路展开的RNN其实向右方向是代表先后顺序（同时也带记忆数据流），RNN-unrolled
跟上下方向意义是不一样的。有没有同学误解那么一排东西是可以同时喂进去的？

batch_size 是每批几串字符串，
num_unrollings 是每串子字符串的长度， unrolled 的意思也是向右展开多少。
（相应的，单个CELL的图，我们成为RNN rolled）

它在初始化的时候先根据 batch_size 把段分好，
然后设立一组游标 _cursor ，是一组哦，不是一个哦！
然后定义好 _last_batch 看或许到哪了。

next() 返回的数据格式，是一个list，list的长度是 num_unrollings+1，
每一个元素，都是一个(batch_size,27)的array，
27是 vocabulary_size，一个27维向量代表一个字符，是one-hot encoding的格式。
(字符串开头还加了上一次获取的最后一个字符，所以实际上字符串长度要比 num_unrollings 多一个）。

"""


class BatchGenerator(object):
    def __init__(self, text, batch_size, num_unrollings):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        segment = self._text_size // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._next_batch()

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""

        """
        假设我们有1000个字符，batch_size = 10
        _next_batch 返回0， 100， 200，300，400，..., 900 位置上的字符
        换而言之，将1000个字符劈成10块，第一个batch返回每一段的第一个字符
        """

        # we will use one-hot-vector, so initialize all with zero is fine
        batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
        for b in range(self._batch_size):
            # b = 0, cursor[0] = 0   ==> 0
            # b = 1, cursor[1] = 100 ==> 100
            # b = 2, cursor[2] = 200 ==> 200
            # self._cursor 是用于存储每个块内的游标，初始化的时候，都是指向每个字符块的首字符
            # self._text[cursor] 取字符， char2id用于得到对应的编号
            # 把对应编号的位置指定为1，就完成了one-hot-vector的设定
            batch[b, char2id(self._text[self._cursor[b]])] = 1.0
            # 每设定完一个字块后，就把游标往后推一位
            # 经过一个循环后，b/cursor的关系如下
            # b = 0, cursor[0] = 1   ==> 1
            # b = 1, cursor[1] = 101 ==> 101
            # b = 2, cursor[2] = 201 ==> 201
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """

        """
        _next_batch 是用于取下一个batch，next则是返回num_unrollings个batches
        如果num_unrolling=3， 则返回4个batches，同时保留最后一个batch，等下一次调next（）的时候
        被保留的batch会放在batches(返回值）的一位。
        """


        batches = [self._last_batch]
        for step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches

"""
再定义两个用来把训练数据转换成可展现字符串的函数。

characters 先从one-hot encoding变回数字，再用id2char变成字符。
batches2string 则将训练数据变成可以展现的字符串。
"""

def characters(probabilities):
    """Turn a 1-hot encoding or probability distribution over the possible
    characters back into its (most likely character representation"""

    return [id2char(c) for c in np.argmax(probabilities, 1)]

def batches2string(batches):
    """Convert a sequence of batches back into their (most likely) string
    representation."""

    """
    从batches中提取一个batch, 每个batch含100个字符，shape = 100 x voc_size
    通过characters()函数，将voc_size这个维度(1-hot)转换成字符，变成长度为100的字符
    经过BatchGenerator切分后，这些字符不是连续的
    现在我们要将batch[0][0], batch[1][0], batch[2][0], ...,batch[batch_size-1][0]
    拼接起来，还原原来的字符串,同样的对 batch[...][1], batch[...][2]....
    
    l = [['a','b','c'], ['d','e','f']]

    s = ['']*len(l[0])
    print(s)
    for l1 in l:
        s = [a+b for a,b in zip(s,l1)]
    print(s)
    
    OUTPUT:
    
    ['', '', '']
    ['ad', 'be', 'cf']
    
    """

    s = [''] * batches[0].shape[0]  # batch_size
    for b in batches:
        # s = [''.join(x) for x in zip(s, characters(b))]
        s = [x+y for x, y in zip(s, characters(b))]
    return s

"""
以下四个函数，在训练中输出摘要时使用

logprob: 用来预测工作完成的如何
crossEntropy = - sum(label * log(prediction)) (from 0 to N)
logprob = cossEntropy/N

后面三个函数 sample_distribution sample random_distribution 是一起使用的。
[random_distribution] 就是生成一个平均分布的，加总和为 1 的 array。
[sample] 则是靠 [sample_distribution] 以传入的 prediction 的概率，随机取一个维设成 1 ，
其他都设成 0 ，也就是按照 prediction 的概率获得一个随机字母。

不清楚为什么要搞这么复杂，是有什么采样的原理我不知道嘛。。。
"""

def logprob(predictions, labels):
    """log-probability of the true labels in a predicted batch"""
    predictions[predictions < 1e-10] = 1e-10

    # np.multiply(list, list) is element wise product

    return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

def sample_distribution(distribution):
    """Sample one element from a distribution assumed to be an array of normalized probabilities"""

    r = random.uniform(0, 1)
    s = 0
    for i in range(len(distribution)):
        s += distribution[i]
        if s>= r:
            return i
    return len(distribution) - 1

def sample(prediction):
    """Trun a (column) prediction into 1-hot encoded samples"""
    p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
    p[0, sample_distribution(prediction[0])] = 1.0
    return p

def random_distribution():
    """Generate a random column of probabiliities"""
    b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
    return b/(np.sum(b, 1).reshape(-1, 1))



if __name__ == "__main__":
    # load raw data
    root = join(expanduser("~"), 'mldata')
    file = join(root, 'text8/text8.zip')
    text = read_data(file)
    print("data size is %d" % len(text))

    # define the first 1000 words as validation, take the rest as training
    valid_size = 1000
    valid_text = text[:valid_size]
    train_text = text[valid_size:]
    train_size = len(train_text)
    print(train_size, train_text[:64])
    print(valid_size, valid_text[:64])


    print(char2id('a'), char2id('z'), char2id(' '), char2id('ï'))
    print(id2char(1), id2char(26), id2char(0))

    batch_size = 64
    num_unrollings = 10


    train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
    valid_batches = BatchGenerator(valid_text, 1, 1)

    print(batches2string(train_batches.next()))
    print(batches2string(train_batches.next()))
    for _ in range(2):
        """
        batch_size=1, 每次返回只含一个batch
        num_unrollings = 1 意味着每一只有一个内容。
        在该实例中可以调用1000次next（）--不重复
        第1001开始重复第一次。
        
        Note:next（）方法会保留前一次的一个字符。
        每个返回都会包含【老字符，新字符】，而新字符在下次返回中变成老字符。
    
        """
        print(batches2string(valid_batches.next()))

    """
    num_nodes 应当是指定，在CELL内有多少个神经元
    """
    num_nodes = 64

    graph = tf.Graph()
    with graph.as_default():

        # Parameters:
        # Input gate: input, previous output, and bias
        ix = tf.Variable(tf.truncated_normal(shape=[vocabulary_size, num_nodes], mean=-0.1, stddev=0.1))
        im = tf.Variable(tf.truncated_normal(shape=[num_nodes, num_nodes], mean=-0.1, stddev=0.1))
        ib = tf.Variable(tf.zeros([1, num_nodes]))
        # Forget gate: input, previous output, and bias
        fx = tf.Variable(tf.truncated_normal(shape=[vocabulary_size, num_nodes], mean=-0.1, stddev=0.1))
        fm = tf.Variable(tf.truncated_normal(shape=[num_nodes, num_nodes], mean=-0.1, stddev=0.1))
        fb = tf.Variable(tf.zeros([1, num_nodes]))
        # Memory cell: input, previous output, and bias
        cx = tf.Variable(tf.truncated_normal(shape=[vocabulary_size, num_nodes], mean=-0.1, stddev=0.1))
        cm = tf.Variable(tf.truncated_normal(shape=[num_nodes, num_nodes], mean=-0.1, stddev=0.1))
        cb = tf.Variable(tf.zeros([1, num_nodes]))
        # Output gate: input, previous output, and bias
        ox = tf.Variable(tf.truncated_normal(shape=[vocabulary_size, num_nodes], mean=-0.1, stddev=0.1))
        om = tf.Variable(tf.truncated_normal(shape=[num_nodes, num_nodes], mean=-0.1, stddev=0.1))
        ob = tf.Variable(tf.zeros([1, num_nodes]))

        # Variable saving state across unrollings
        """
        saved_output 是向上的产出，saved_state 是自己的状态记忆。
        """
        saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
        saved_state  = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)

        # Classifer weights and bias
        w = tf.Variable(tf.truncated_normal(shape=[num_nodes, vocabulary_size], mean=-0.1, stddev=0.1))
        b = tf.Variable(tf.zeros([vocabulary_size]))


        # define cell computation
        def lstm_cell(i, o, state):
            """
            Create LSTM cell,
            See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf

            Note: in this formulation, we omit the various connections between the previous state and the gates
            在论文里为每个门多加了一个控制变量， C(t-1), 本来C变量作为中间变量，外部看不见，值是参与输出是才和O(t)点积，
            而O(t)已经间接参与了门控制，论文里为此再加入门控制的参数，原因不是很明确。是否所有的lstm都是从这一个点出发的，也未知。

            该实现却是和colah的介绍一致。

            """

            """
            另外，我理解为每一个CELL内都有num_nodes神经元，输出也是这些个神经元
            所以即便只有一个CELL,其实表现力也是很强的。
            """

            input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
            forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
            update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
            state = forget_gate * state + input_gate*tf.tanh(update)
            output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
            return output_gate * tf.tanh(state), state


        # input data
        """
        一下的这种赋值方法是要展示另一种可能性吗？
        为什么不是直接用3维数组？可能是要体现一个一个输入。。。
        """

        # TODO: Chage to 3-dim array
        train_data = list()
        for _ in range(num_unrollings + 1):
            train_data.append(
                tf.placeholder(tf.float32, shape=[batch_size, vocabulary_size])
            )
        train_inputs = train_data[:num_unrollings]
        train_labels = train_data[1:]  # labels are inputs shifted by one time step

        # unrolled LSTM loop
        """
        这里也是一个 batch 同时处理的。但为了容易理解，我先假设 batch_size=1 ，
        然后假设我们要训练一个字符串 abcdefg,
        那么 train_inputs 是 abcdef，
        train_labels 是 bcdefg 。
        
        根据前面定义变量的时候规定，初始 saved_output 和 saved_state 都是全零。
        依次输入 a b c d e f ，把每一次的输出放在一起形成一个 list 就是 outputs。
        """
        outputs = list()
        output  = saved_output
        state = saved_state
        for i in train_inputs:
            output, state = lstm_cell(i, output, state)
            outputs.append(output)

        # State saving across unrollings

        """
        因为不是顺序执行语言，一般模型如果不是相关的语句，其执行是没有先后顺序的，
        control_dependencies 的作用就是建立先后顺序，保证前面两句被执行后，才执行后面的内容。
        这里也就是先把 saved_output 和 saved_state 保存之后，再计算 logits 和 loss。
        否则因为下面计算时没有关联到 saved_output 和 saved_state，
        如果不用 control_dependencies 那上面两句保存就不会被优化语句触发。
        """

        with tf.control_dependencies([saved_output.assign(output), saved_state.assign(state)]):
            # Classifer
            logits = tf.nn.xw_plus_b(tf.concat(outputs, 0), w, b)

            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.concat(train_labels, 0), logits=logits
                )
            )

            """
            tf.concat(values, 0) 是指在 0 维上把 values 连接起来。
            本来 outputs 是一个 list，每一个元素都是一个27维向量表示一个字母（还是假设 batch_size=1）
            通过 tf.concat 把结果连接起来，成为一个向量，可以拿来乘以 w 加上 b 
            这样进入一个 full connection，从而得到 logits
            """

        # Optimizer
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(learning_rate=10.0,
                                                   global_step=global_step,
                                                   decay_steps=5000,
                                                   decay_rate=0.1,
                                                   staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        gradients, v = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
        optimizer = optimizer.apply_gradients(
            zip(gradients, v), global_step=global_step
        )

        """
        tf.train.exponential_decay 可以用来实现 learning_rate 的指数型衰减，
        越到后面 learning_rate 越小。（依赖后面修改 global_step 值来实现）

        optimizer 定义成使用标准 Gradient Descent 。每一种 optimizer 都有几个标准接口，
        我们前面常用的是 minimize 接口，他自动的调整整个 Graph 中可调节的 Variables 尝试最小化 loss。
        其实 minimize 函数就是这两步并起来： compute_gradients 和 apply_gradients。
        先计算梯度值，然后再把那些参数减去梯度值。
        这里把两步分开了，为了在 apply 之前先处理一下梯度值，

        compute_gradients 函数返回一个list，里面是一对一对的 gradient 和 variable，
        说明针对某个可调整的变量，他的梯度是多少。
        
        clip_by_global_norm 避免梯度值过大产生 Exploding Gradients 梯度爆炸问题
        Deep Learning 视频里有过这么一段描述，当 delta_W 大于某个门限值得时候，delta_max
        我们通过这个公式计算delta_W
                                        delta_max
        delta_W = delta_W * --------------------------------
                             max(abs(delta_W), delt_max)
        
        clip_by_global_norm 的具体计算是，先计算 global_norm ，也就是整个 tensor 的模（二范数）。
        看这个模是否大于文中的1.25，如果大于，则结果等于 gradients * 1.25 / global_norm，如果不大于，就不变。
        
        最后，apply_gradients。这里传入的 global_step 是会被修改的，
        每次加一，这样下次计算 learning_rate 的时候就会使用新的 global_step 值。
        
        """

        # predictions.
        train_prediction = tf.nn.softmax(logits)

        # Sampling and validation eval: batch 1, no unrolling.
        # sample_input 是一个1-hot编码后的字符  ==> 经过同样的LSTM CELL得到下一个预测的字符 sample_prediction
        sample_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size])
        saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
        saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
        sample_output, sample_state = lstm_cell(
            sample_input, saved_sample_output, saved_sample_state)
        with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                      saved_sample_state.assign(sample_state)]):
            sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))

        reset_sample_state = tf.group(
            saved_sample_output.assign(tf.zeros([1, num_nodes])),
            saved_sample_state.assign(tf.zeros([1, num_nodes])))


    num_steps = 7001
    summary_frequency = 100

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print('Initialized')
        mean_loss = 0
        for step in range(num_steps):
            batches = train_batches.next()
            feed_dict = dict()
            for i in range(num_unrollings + 1):
                """
                这是一组placeholder, 这样赋值就把一个系列的数据全部输入了
                如果从keras的参数角度来看就是一个[batch, unrolling, 1-hot-vector]的输入
                （同时也定义了输入[：num_unrolling],和输出[1:] 
                """
                feed_dict[train_data[i]] = batches[i]
            _, l, predictions, lr = session.run(
                [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
            """
            到这里，网络训练结束了，接下来就是验证的工作。
            """

            mean_loss += l
            if step % summary_frequency == 0:
                if step > 0:
                    mean_loss = mean_loss / summary_frequency
                # The mean loss is an estimate of the loss over the last few batches.
                print(
                    'Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
                mean_loss = 0

                """
                predictions 是outputs 的softmax（Logit）
                一个output是一个24长的向量
                labels 是 batch*num_unrolling*vol_size的一位向量
                """

                labels = np.concatenate(list(batches)[1:])


                print('Minibatch perplexity: %.2f' % float(
                    np.exp(logprob(predictions, labels))))

                """
                随机起一个字符，然后持续利用LSTM计算下一个字符。并把这个字符作为输入，然后做下一次计算
                sample_prediction利用了训练好的lstm cell，没做完一个句子，reset lstm 的state, output
                
                第一次的生成
                ================================================================================
                utrxaaoqiaqnoaueb qnd imuuineuyoctfiumemx olh edri hironcja ppfnjirsgna el pulo 
                ehkuirmf avphiautpriefsnepyztytoze xn cnrtnuww  aw h qtqmgemrmym iseismulfspi at
                yftk ewg  a slhrerxf     skdvozufoqxp  cebec aai hvcarjulavxq nqvyfhgc vzoo svot
                xm b urvddrn tceilurd es on ivxsg ps ktwry e vejyd eratpazbggwjnejrmbhrpaa qrkpo
                hdduwdx rjtmvkcofprttnxi orofepxdh hdvl dtde dnyh qvujhmsiy nosmrpzjhsoti lsejet
                ================================================================================                
                
                最后一次的生成
                
                ================================================================================
                ts where in proent sming sont of ama foundry handwere eventaph international dem
                and wild one six rerueing negentish there shem lame malaece for and mistor to ha
                page kabarech have traystlert serviers are pinott cettury and altoopimed pats as
                heraus under thoodor be denine are actic histill still allow decampled populate 
                velical pier surn in the lusling rations of country the first the begived sameni
                ================================================================================
                
                """

                if step % (summary_frequency * 10) == 0:
                    # Generate some samples.
                    print('=' * 80)
                    for _ in range(5):
                        feed = sample(random_distribution())
                        sentence = characters(feed)[0]
                        reset_sample_state.run()
                        for _ in range(79):
                            prediction = sample_prediction.eval({sample_input: feed})

                            """
                            这个地方值得写一下。
                            我们按照预测值计算得出概率分布，然后我们并不按照最高的那个来选择，而是通过采样才完成选择。
                            这是常常忽略的地方。
                            """

                            feed = sample(prediction)

                            """
                            sentence 是按照采样得到的字符加入，并作为下一个的输入。
                            """

                            sentence += characters(feed)[0]
                        print(sentence)
                    print('=' * 80)


                # Measure validation set perplexity.
                reset_sample_state.run()
                valid_logprob = 0
                for _ in range(valid_size):
                    b = valid_batches.next()
                    predictions = sample_prediction.eval({sample_input: b[0]})
                    valid_logprob = valid_logprob + logprob(predictions, b[1])
                    """
                    predictions = [24]
                    b = [2, 24]
                    logprob 本来是为了直接计算perplexity 设计的，但是到了这里变成一个一个的计算P(S_i)
                    然后取平均数，由于设计的最初设想要取平均数，所以子程序做了改动，除以了b[1].shape[0] 
                    这个值一致都等于1。
                    
                    
                    解释一下perplexity: 该参数是用于衡量show well a probability distribution or probability model predicts a sample.
                    公式为2^(-l), l = mean(logP(S_i)) log是以2为底
                    
                    该程序中，用label和prediction做elementwise-product, 取得真实值得估算概率，忽略掉其他的概率。然后累加
                    """

                print('Validation set perplexity: %.2f' % float(np.exp(
                    valid_logprob / valid_size)))


"""
Note:

In [15]: np.argmax(a, 1)
Out[15]: array([0, 1, 1, 2, 2])

In [16]: a
Out[16]:
array([[ 0.96281771,  0.94948446,  0.43436422],
       [ 0.78742128,  0.87818258,  0.86642704],
       [ 0.29913411,  0.74497496,  0.52710406],
       [ 0.20787206,  0.4585659 ,  0.95117295],
       [ 0.87954721,  0.43695005,  0.90539801]])

Note:

TF assign, variable 赋值
In [34]: s.run(a.assign([1]))
Out[34]: array([ 1.], dtype=float32)

In [35]: s.run(a)
Out[35]: array([ 1.], dtype=float32)

In [36]: s.run(a.assign([2]))
Out[36]: array([ 2.], dtype=float32)

In [37]: s.run(a)
Out[37]: array([ 2.], dtype=float32) 


Note:

tf.concat(value, 0)

In [42]: x
Out[42]: [[1, 2, 3], [2, 3, 4]]

In [43]: s.run(tf.concat(x, 0))
Out[43]: array([1, 2, 3, 2, 3, 4], dtype=int32)


"""