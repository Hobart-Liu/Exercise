import numpy as np
import pandas as pd

def my_strategy(hist_list, threshold1, threshold2):
    '''
    按照两个阈值来决定投接下来投1还是0
    :param hist_list: 观察列表
    :param threshold1: 阈值：1/0 比 0/1 多出多少
    :param threshold2: 阈值：最后至少重复出现多少个相同的数字
    :return:
    '''
    diff = sum(hist_list) - (len(hist_list) - sum(hist_list))
    # 计算最后的出现的重复的数字
    last_repeat = 0
    for i in range(1, len(hist_list)):
        s = set(hist_list[-i:])
        if len(s) == 1:
            last_repeat = i
        else:
            break

    # 如果出现1/0 的数目远大于0/1的情况，且最后几个数字相同的话
    if (abs(diff) > threshold1) and (last_repeat >= threshold2):
        # 1 >> 0, 且最后的重复数字是1
        if (diff > 0) and hist_list[-1] == 1:     # number of '1' > number of '0'
            return 0
        # 0 >> 1, 且最后的重复数字是0
        elif (diff < 0) and hist_list[-1] == 0:    # number of '0' > number of '1'
            return 1
        else:
            return None
    else:
        return None


class Dice():
    def __init__(self, total_sim =1000, observation = 100):
        self.sim = np.random.binomial(1, 0.5, total_sim)
        self.length = total_sim
        self.p_start = -1
        self.p_end = observation - 1

    def has_data(self):
        return self.p_end < self.length - 1


    def next(self):
        if self.has_data() is False:
            return None
        else:
            self.p_start += 1
            self.p_end += 1
            return dict(obs = self.sim[self.p_start:self.p_end], next=self.sim[self.p_end])

    def getlist(self):
        return self.sim

n_largers = [5, 10, 15, 20, 25, 30]
n_repeats = [3,4,5,6,7,8,9,10]

dices = 1
summary = np.zeros(shape=(len(n_repeats), len(n_largers)))
print(summary.shape)

for k in range(dices):  # let's play 1000 rounds
    print(k)
    simulations = Dice(1000, 100)
    myprofit = np.zeros(shape=(len(n_repeats), len(n_largers)))
    while(simulations.has_data()):
        data = simulations.next()
        for i, n_large in enumerate(n_largers):
            for j, n_repeat in enumerate(n_repeats):
                myaction = my_strategy(data['obs'], n_large, n_repeat)
                if myaction is None:
                    pass
                elif myaction == data['next']:
                    myprofit[j][i] += 1
                else:
                    myprofit[j][i] -= 1
    print(myprofit)
    summary += myprofit


print(summary)


