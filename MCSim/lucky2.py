import numpy as np
import pandas as pd

# def my_strategy(hist_list, threshold):
#     p = sum(hist_list)/len(hist_list)
#     ret = None
#     if abs(p-0.5) >= threshold:
#         ret = np.random.binomial(1, 1-p, 1)
#     return ret

def my_strategy(hist_list, threshold):
    p = sum(hist_list)/len(hist_list)
    ret = None
    if abs(p-0.5) >= threshold:
        p = 1 - p/3
        avg = np.average(np.random.binomial(1, p, 20))
        if avg < p : ret = 1
        else:        ret = 0
    return ret


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


dices = 1000
thresholds = [ 0.3, 0.4]
summary = np.zeros(len(thresholds))

for k in range(dices):
    simulations = Dice(100, 20)
    myprofit = np.zeros(len(thresholds))
    actions = np.zeros(len(thresholds))
    while(simulations.has_data()):
        data = simulations.next()
        for i, t in enumerate(thresholds):
            myaction = my_strategy(data['obs'], t)
            if myaction is not None:
                actions[i] += 1
                if myaction == data['next']:
                    myprofit[i] += 1
                else:
                    myprofit[i] -= 1

    print("played {} rounds, profit is {}".format(actions, myprofit))
    summary += myprofit

print(summary)

