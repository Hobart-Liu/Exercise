import random
import matplotlib.pyplot as plt
import numpy as np

def calculate_score(candidates, stop_time, ratio_threshold):
    for i in range(stop_time, len(candidates)):
        observations = np.array(candidates[:i])
        threshold = np.percentile(observations, 100-ratio_threshold)
        if candidates[i] >= threshold or i == len(candidates) - 1:
            return candidates[i]

lifes = [[random.uniform(0, 1) for i in range(20)] for i in range(1000)]
strategies = [(stop_time, ratio_threshold) for stop_time in range(1, 20) for ratio_threshold in range(1, 20, 1)]
scores = [0 for i in range(len(strategies))]
print(len(scores))
for i in range(len(strategies)):
    print(i)
    for life in lifes:
        scores[i] += calculate_score(life, strategies[i][0], strategies[i][1])
index_max = np.argmax(scores)
print(strategies[index_max])

#### seaborn
import seaborn as sns

x = list(range(1, 20))
y = list(range(1, 20))
the_score = np.array(scores).reshape(len(x), len(y))
the_score = np.rint(the_score/10)
ax = sns.heatmap(the_score.T, annot=True, linewidths=.5, cmap="YlGnBu", cbar=False )
ax.set_xlabel("Stop Time")
ax.set_ylabel("Threshold")
plt.show()



exit()

max_score = max(scores)
min_score = min(scores)

colors = [1-((scores[i] - min_score) / (max_score - min_score) * 0.9 + 0.1) for i in range(len(scores))]

for i in range(len(scores)):
    plt.plot(strategies[i][0], strategies[i][1], 'o', color = str(colors[i]), markersize = (1-colors[i])*20 )
plt.plot(strategies[index_max][0], strategies[index_max][1], 'ro', markersize = (1-colors[index_max])*20 )
plt.xlim([-1, 20])
plt.ylim([-1, 20])
plt.xlabel('stop time')
plt.ylabel('percentage')

plt.show()




