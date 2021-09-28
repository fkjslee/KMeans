import numpy as np
import matplotlib.pyplot as plt
import torch


def generate_data():
    np.random.seed(1234)
    return [[np.random.random(1)[0] * 100, np.random.random(1)[0] * 100] for _ in range(100)]


data = generate_data()
data = np.float32(data)
k = 3
clusters = [{"center": data[i], "set": [data[i]]} for i in range(k)]

plt.plot(data[:, 0], data[:, 1], 'o')
plt.title("original")
plt.show()

for epoch in range(10):
    for u in range(k):
        cluster = np.float32(clusters[u]['set'])
        plt.plot(cluster[:, 0], cluster[:, 1], 'o')
        plt.plot(clusters[u]['center'][0], clusters[u]['center'][1], '*')
        plt.title("epoch {}".format(epoch))
    for u in range(k):
        clusters[u]['set'] = []
    for p in data:
        dist = []
        for u in range(k):
            dist.append(np.sum(np.square(clusters[u]['center'] - p)))
        clusters[np.argmin(dist)]['set'].append(p)
    for u in range(k):
        cluster = np.float32(clusters[u]['set'])
        clusters[u]['center'] = np.float32([np.mean(cluster[:, 0]), np.mean(cluster[:, 1])])
    plt.show()
