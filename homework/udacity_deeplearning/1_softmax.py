import numpy as np
import matplotlib.pyplot as plt

def softmax(x, sum_axis=0):
    '''
    softmax 实现
    :param x: list or np.array (can be n*m table)
    :return: np.array of softmax
    '''
    p = np.exp(x)
    total = np.sum(p, axis=sum_axis)
    return np.array(p/total)

def task1():
    # 简单的测试
    score = [3.0, 1.0, 0.2]
    print(softmax(score))

    # 产生一系列的数据，x取值从-2.0到6，步进为0.1
    # np.exp(x)会有所不一样，随x有负到正，(x, 1, 0.2x)的占比不同
    x = np.arange(-2.0, 6.0, 0.1)
    print("We generated {} X as input".format(len(x)))

    '''
    In [19]: x
    Out[19]: array([10, 20, 30, 40, 50])

    In [20]: np.vstack([x, x*2])
    Out[20]:
    array([[ 10,  20,  30,  40,  50],
           [ 20,  40,  60,  80, 100]])
           
    In [21]: sum(np.vstack([x, x*2]))
    Out[21]: array([ 30,  60,  90, 120, 150])
    
    每个垂直方向认为是一条记录, vector. 
    np.vstack 方法用于批量产生vector值得借鉴一下。
    softmax  是做垂直方向的相加。
    
    '''

    fig, axs = plt.subplots(3, 1, sharex='all', sharey='all', figsize=(5, 8))


    scores = np.vstack([x, np.ones_like(x), 0.2*np.ones_like(x)])
    s = softmax(scores)
    # 图中可以看出，1， 0.2 是常量，但是随着exp(x)的增加，softmax后的比重降低
    axs[0].plot(x, s[0], linewidth=2, label='x')
    axs[0].plot(x, s[1], linewidth=2, label='1')
    axs[0].plot(x, s[2], linewidth=2, label='0.2')
    axs[0].title.set_text("Normal")
    axs[0].legend()

    # Note: y.T 是可以同时画多条曲线
    # plt.plot(x, softmax(scores).T, linewidth=2)

    # 放大scores * 10
    # 差异变大，经过放大后的softmax，各个值得占比差异显著
    scores_larger = scores * 10
    s = softmax(scores_larger)
    axs[1].plot(x, s[0], linewidth=2, label='x')
    axs[1].plot(x, s[1], linewidth=2, label='1')
    axs[1].plot(x, s[2], linewidth=2, label='0.2')
    axs[1].title.set_text('X 10')
    axs[1].legend()

    # 缩小scores /10
    # 趋向平均，各个值经过softmax后差别不大
    scores_smaller = scores / 10
    s = softmax(scores_smaller)
    axs[2].plot(x, s[0], linewidth=2, label='x')
    axs[2].plot(x, s[1], linewidth=2, label='1')
    axs[2].plot(x, s[2], linewidth=2, label='0.2')
    axs[2].title.set_text('/10')
    axs[2].legend()


    plt.show()



