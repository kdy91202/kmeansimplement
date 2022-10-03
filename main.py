# created by 김대엽 2020253113
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import random


"""def main(args):
    with open(args, 'r') as file:
        print(file.read())
    
if __name__ == '__main__':
    main(sys.argv[1])"""


data1 = []
data2 = []
data3 = []
data4 = []
data5 = []
data6 = []
data7 = []
data8 = []
data9 = []
data10 = []
coldata1 = []
coldata2 = []
coldata3 = []
coldata4 = []
coldata5 = []
coldata6 = []
coldata7 = []
coldata8 = []
coldata9 = []
coldata10 = []

num = -1

with open('assignment2_input.txt', 'r') as file:
    for line in file:
        num = num + 1
        if num <= 49:
            data1.append(line.strip().split('\t'))
        if num >= 50 and num <= 99:
            data2.append(line.strip().split('\t'))
        if num >= 100 and num <= 149:
            data3.append(line.strip().split('\t'))
        if num >= 150 and num <= 199:
            data4.append(line.strip().split('\t'))
        if num >= 200 and num <= 249:
            data5.append(line.strip().split('\t'))
        if num >= 250 and num <= 299:
            data6.append(line.strip().split('\t'))
        if num >= 300 and num <= 349:
            data7.append(line.strip().split('\t'))
        if num >= 350 and num <= 399:
            data8.append(line.strip().split('\t'))
        if num >= 400 and num <= 449:
            data9.append(line.strip().split('\t'))
        if num >= 450 and num <= 500:
            data10.append(line.strip().split('\t'))


for i in range(12) :
    cols = np.array(data1).T[i]
    coldata1.append(cols)
    coldatac1 = np.asarray(coldata1, dtype=float)
    cols = np.array(data2).T[i]
    coldata2.append(cols)
    cols = np.array(data3).T[i]
    coldata3.append(cols)
    cols = np.array(data4).T[i]
    coldata4.append(cols)
    cols = np.array(data5).T[i]
    coldata5.append(cols)
    cols = np.array(data6).T[i]
    coldata6.append(cols)
    cols = np.array(data7).T[i]
    coldata7.append(cols)
    cols = np.array(data8).T[i]
    coldata8.append(cols)
    cols = np.array(data9).T[i]
    coldata9.append(cols)
    cols = np.array(data10).T[i]
    coldata10.append(cols)

'''for i in range(12):
    sb.scatterplot(x=coldata1[i], y=coldata2[i])
    sb.scatterplot(x=coldata3[i], y=coldata4[i])
    sb.scatterplot(x=coldata5[i], y=coldata6[i])
    sb.scatterplot(x=coldata7[i], y=coldata8[i])
    sb.scatterplot(x=coldata9[i], y=coldata10[i])


    plt.scatter(coldata1[i], coldata2[i])
    plt.scatter(coldata3[i], coldata4[i])
    plt.scatter(coldata5[i], coldata6[i])
    plt.scatter(coldata7[i], coldata8[i])
    plt.scatter(coldata9[i], coldata10[i])

plt.show()'''

'''
coldatac1 = StandardScaler().fit_transform(coldatac1)

for i in range(12) :
    sb.scatterplot(x=[X[i] for X in coldatac1],
                 y=[X[i+1] for X in coldatac1],
                 legend=None)

plt.xlabel("x")
plt.ylabel("y")
plt.show()
'''


def euclidist(x, y) :
    return np.round(np.sqrt(np.sum((x - y)**2, axis=1)), 3)

class KMeans :

    def __init__(self, clusters=10, max_iter=50):
        self.clusters = clusters
        self.max_iter = max_iter

    def fit(self, xtrain):
        mins, maxs = np.min(xtrain, axis=0), np.max(xtrain, axis=0)
        self.centroids = [random.uniform(mins, maxs) for _ in range(self.clusters)]

        iters = 0
        prev_centroids = None

        while np.not_equal(self.centroids, prev_centroids).any() and iters < self.max_iter :
            sorted_pts = [[] for _ in range(self.clusters)]
            for x in xtrain :
                dist = euclidist(x, self.centroids)
                centroid_idx = np.argmin(dist)
                sorted_pts[centroid_idx].append(x)

                prev_centroids = self.centroids
                self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_pts]

                for i, centroid in enumerate(self.centroids) :
                    if np.isnan(centroid).any() :
                        self.centroids[i] = prev_centroids[i]
                    iters += 1

    def evaluate(self, X):
        centeroids = []
        centeroids_idx = []
        for x in X :
            dists = euclidist(x, self.centroids)
            centeroids_idx = np.argmin(dists)
            centeroids.append(self.centeroids_idx)

        return centeroids, centeroids_idx


kmeans = KMeans(clusters=10)
kmeans.fit(coldatac1)

class_centers, classification = kmeans.evaluate(coldatac1)

for i in range(12):
    sb.scatterplot(x=[X[i] for X in coldatac1],
                   y=[X[i + 1] for X in coldatac1],
                   style=classification,
                   legend=None)
    plt.plot([x for x, _ in kmeans.centroids],
             [y for _, y in kmeans.centroids],
             '+',
             markersize=10,
             )
plt.show()

