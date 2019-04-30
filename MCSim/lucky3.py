import numpy as np


class Dice():
    def __init__(self, total_sim = 1000, observation = 100):
        self.sim = np.random.binomial(1, 0.5, total_sim)
        self.length = total_sim
        self.p_obs_start = 0
        self.p_obs_end = observation
        self.shift = observation * 2
        self.p_test_start = observation
        self.p_test_end = observation * 2

    def has_data(self):
        if len(self.sim) < self.p_test_end:
            return False
        else:
            return True

    def next(self):
        if self.has_data():
            ret = dict(data=self.sim[self.p_obs_start: self.p_obs_end],
                       test=self.sim[self.p_test_start: self.p_test_end])

            self.p_obs_start += self.shift
            self.p_obs_end += self.shift
            self.p_test_start += self.shift
            self.p_test_end += self.shift

            return ret

        else:
            return None


def my_strategy(hist_list, threshold):
    p = sum(hist_list)/len(hist_list)

    ret = None

    if abs(p - 0.5) >= threshold:
        ret = np.random.binomial(1, 1-p, len(hist_list))

    return ret

dices = 1000
thresholds = [0.1, 0.2, 0.3, 0.4]
summary = None
myspend = 0
myobs = 20

for k in range(dices):
    simulations = Dice(120, myobs)
    myprofit = np.zeros(len(thresholds))
    while(simulations.has_data()):
        d = simulations.next()
        for i , t in enumerate(thresholds):
            myactions = my_strategy(d['data'], t)
            if myactions is not None:
                sum_up = sum(myactions == d['test'])/len(myactions)
                myprofit[i] = sum_up


    if summary is None:
        summary = myprofit
    else:
        summary = np.vstack((summary, myprofit))

print(np.average(summary, axis=0))





















