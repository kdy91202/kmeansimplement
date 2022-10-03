# created by 김대엽 2020253113
import numpy as np
import random
import sys

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
result1 = []
result2 = []
result3 = []
result4 = []
result5 = []
result6 = []
result7 = []
result8 = []
result9 = []
result10 = []

nu = -1
nu2 = 49
nu3 = 100
nu4 = 150
nu5 = 200
nu6 = 250
nu7 = 300
nu8 = 350
nu9 = 400
nu10 = 450
num = -1


def main(args):
    with open(args, 'r') as file:
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

if __name__ == '__main__':
    main(sys.argv[1])

datac1 = np.asarray(data1, dtype=float)
datac2 = np.asarray(data2, dtype=float)
datac3 = np.asarray(data3, dtype=float)
datac4 = np.asarray(data4, dtype=float)
datac5 = np.asarray(data5, dtype=float)
datac6 = np.asarray(data6, dtype=float)
datac7 = np.asarray(data7, dtype=float)
datac8 = np.asarray(data8, dtype=float)
datac9 = np.asarray(data9, dtype=float)
datac10 = np.asarray(data10, dtype=float)


def euclidist(x, y) :
    result = np.round(np.sqrt(np.sum((x - y)**2, axis=1)), 3)
    return result

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
        centroids = []
        centroid_idxs = []
        for x in X:
            dists = euclidist(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)

        return centroids, centroid_idxs, centroid_idx


kmeans1 = KMeans()
kmeans2 = KMeans()
kmeans3 = KMeans()
kmeans4 = KMeans()
kmeans5 = KMeans()
kmeans6 = KMeans()
kmeans7 = KMeans()
kmeans8 = KMeans()
kmeans9 = KMeans()
kmeans10 = KMeans()
kmeans1.fit(datac1)
kmeans2.fit(datac2)
kmeans3.fit(datac3)
kmeans4.fit(datac4)
kmeans5.fit(datac5)
kmeans6.fit(datac6)
kmeans7.fit(datac7)
kmeans8.fit(datac8)
kmeans9.fit(datac9)
kmeans10.fit(datac10)

class_centers1, classification1, centers1 = kmeans1.evaluate(datac1)
class_centers2, classification2, centers2 = kmeans2.evaluate(datac2)
class_centers3, classification3, centers3 = kmeans3.evaluate(datac3)
class_centers4, classification4, centers4 = kmeans4.evaluate(datac4)
class_centers5, classification5, centers5 = kmeans5.evaluate(datac5)
class_centers6, classification6, centers6 = kmeans6.evaluate(datac6)
class_centers7, classification7, centers7 = kmeans7.evaluate(datac7)
class_centers8, classification8, centers8 = kmeans8.evaluate(datac8)
class_centers9, classification9, centers9 = kmeans9.evaluate(datac9)
class_centers10, classification10, centers10 = kmeans10.evaluate(datac10)

#1~50
for i in range(50) :
    nu = nu + 1
    if nu <= 49:
        if classification1[i] == 0 :
            result1.append(nu)
        if classification1[i] == 1 :
            result2.append(nu)
        if classification1[i] == 2 :
            result3.append(nu)
        if classification1[i] == 3 :
            result4.append(nu)
        if classification1[i] == 4:
            result5.append(nu)
        if classification1[i] == 5:
            result6.append(nu)
        if classification1[i] == 6:
            result7.append(nu)
        if classification1[i] == 7:
            result8.append(nu)
        if classification1[i] == 8:
            result9.append(nu)
        if classification1[i] == 9:
            result10.append(nu)
#51~100
for i in range(50):
    nu2 = nu2 + 1
    if nu2 >= 50 and nu2 <= 99:
        if classification2[i] == 0 :
            result1.append(nu2)
        if classification2[i] == 1 :
            result2.append(nu2)
        if classification2[i] == 2 :
            result3.append(nu2)
        if classification2[i] == 3 :
            result4.append(nu2)
        if classification2[i] == 4:
            result5.append(nu2)
        if classification2[i] == 5:
            result6.append(nu2)
        if classification2[i] == 6:
            result7.append(nu2)
        if classification2[i] == 7:
            result8.append(nu2)
        if classification2[i] == 8:
            result9.append(nu2)
        if classification2[i] == 9:
            result10.append(nu2)

#101~150
for i in range(50):
    nu3 = nu3 + 1
    if nu3 >= 100 and nu3 <= 149:
        if classification3[i] == 0 :
            result1.append(nu3)
        if classification3[i] == 1 :
            result2.append(nu3)
        if classification3[i] == 2 :
            result3.append(nu3)
        if classification3[i] == 3 :
            result4.append(nu3)
        if classification3[i] == 4:
            result5.append(nu3)
        if classification3[i] == 5:
            result6.append(nu3)
        if classification3[i] == 6:
            result7.append(nu3)
        if classification3[i] == 7:
            result8.append(nu3)
        if classification3[i] == 8:
            result9.append(nu3)
        if classification3[i] == 9:
            result10.append(nu3)
#151~200

for i in range(50):
    nu4 = nu4 + 1
    if nu4 >= 150 and nu4 <= 199:
        if classification4[i] == 0 :
            result1.append(nu4)
        if classification4[i] == 1 :
            result2.append(nu4)
        if classification4[i] == 2 :
            result3.append(nu4)
        if classification4[i] == 3 :
            result4.append(nu4)
        if classification4[i] == 4:
            result5.append(nu4)
        if classification4[i] == 5:
            result6.append(nu4)
        if classification4[i] == 6:
            result7.append(nu4)
        if classification4[i] == 7:
            result8.append(nu4)
        if classification4[i] == 8:
            result9.append(nu4)
        if classification4[i] == 9:
            result10.append(nu4)
#201~250

for i in range(50):
    nu5 = nu5 + 1
    if nu5 >= 200 and nu5 <= 249:
        if classification5[i] == 0 :
            result1.append(nu5)
        if classification5[i] == 1 :
            result2.append(nu5)
        if classification5[i] == 2 :
            result3.append(nu5)
        if classification5[i] == 3 :
            result4.append(nu5)
        if classification5[i] == 4:
            result5.append(nu5)
        if classification5[i] == 5:
            result6.append(nu5)
        if classification5[i] == 6:
            result7.append(nu5)
        if classification5[i] == 7:
            result8.append(nu5)
        if classification5[i] == 8:
            result9.append(nu5)
        if classification5[i] == 9:
            result10.append(nu5)

#251~300
for i in range(50):
    nu6 = nu6 + 1
    if nu6 >= 250 and nu6 <= 299:
        if classification6[i] == 0 :
            result1.append(nu6)
        if classification6[i] == 1 :
            result2.append(nu6)
        if classification6[i] == 2 :
            result3.append(nu6)
        if classification6[i] == 3 :
            result4.append(nu6)
        if classification6[i] == 4:
            result5.append(nu6)
        if classification6[i] == 5:
            result6.append(nu6)
        if classification6[i] == 6:
            result7.append(nu6)
        if classification6[i] == 7:
            result8.append(nu6)
        if classification6[i] == 8:
            result9.append(nu6)
        if classification6[i] == 9:
            result10.append(nu6)
#301~350

for i in range(50):
    nu7 = nu7 + 1
    if nu7 >= 300 and nu7 <= 349:
        if classification7[i] == 0 :
            result1.append(nu7)
        if classification7[i] == 1 :
            result2.append(nu7)
        if classification7[i] == 2 :
            result3.append(nu7)
        if classification7[i] == 3 :
            result4.append(nu7)
        if classification7[i] == 4:
            result5.append(nu7)
        if classification7[i] == 5:
            result6.append(nu7)
        if classification7[i] == 6:
            result7.append(nu7)
        if classification7[i] == 7:
            result8.append(nu7)
        if classification7[i] == 8:
            result9.append(nu7)
        if classification7[i] == 9:
            result10.append(nu7)
#351~400
for i in range(50):
    nu8 = nu8 + 1
    if nu8 >= 350 and nu8 <= 399:
        if classification8[i] == 0 :
            result1.append(nu8)
        if classification8[i] == 1 :
            result2.append(nu8)
        if classification8[i] == 2 :
            result3.append(nu8)
        if classification8[i] == 3 :
            result4.append(nu8)
        if classification8[i] == 4:
            result5.append(nu8)
        if classification8[i] == 5:
            result6.append(nu8)
        if classification8[i] == 6:
            result7.append(nu8)
        if classification8[i] == 7:
            result8.append(nu8)
        if classification8[i] == 8:
            result9.append(nu8)
        if classification8[i] == 9:
            result10.append(nu8)
#401~450
for i in range(50):
    nu9 = nu9 + 1
    if nu9 >= 400 and nu9 <= 449:
        if classification9[i] == 0 :
            result1.append(nu9)
        if classification9[i] == 1 :
            result2.append(nu9)
        if classification9[i] == 2 :
            result3.append(nu9)
        if classification9[i] == 3 :
            result4.append(nu9)
        if classification9[i] == 4:
            result5.append(nu9)
        if classification9[i] == 5:
            result6.append(nu9)
        if classification9[i] == 6:
            result7.append(nu9)
        if classification9[i] == 7:
            result8.append(nu9)
        if classification9[i] == 8:
            result9.append(nu9)
        if classification9[i] == 9:
            result10.append(nu9)
#451~500
for i in range(50):
    nu10 = nu10 + 1
    if nu10 >= 450 and nu10 <= 500:
        if classification10[i] == 0 :
            result1.append(nu10)
        if classification10[i] == 1 :
            result2.append(nu10)
        if classification10[i] == 2 :
            result3.append(nu10)
        if classification10[i] == 3 :
            result4.append(nu10)
        if classification10[i] == 4:
            result5.append(nu10)
        if classification10[i] == 5:
            result6.append(nu10)
        if classification10[i] == 6:
            result7.append(nu10)
        if classification10[i] == 7:
            result8.append(nu10)
        if classification10[i] == 8:
            result9.append(nu10)
        if classification5[i] == 9:
            result10.append(nu10)

sys.stdout = open("assignment2_output.txt", 'w')

print('0:', result1)
print('1:', result2)
print('2:', result3)
print('3:', result4)
print('4:', result5)
print('5:', result6)
print('6:', result7)
print('7:', result8)
print('8:', result9)
print('9:', result10)

