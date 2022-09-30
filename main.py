# created by 김대엽 2020253113
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import sys

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

def euclidist(x, y) :
    return round(np.sqrt(np.sum(x - y)**2), 4)

def gcenter(x) :
    x = np.array(x)
    return x.mean(axis=0)

def cluster(x, k=10, iteration=10) :
    log = []
    centers = x[np.random.choice(len(x), size=k, replace=False)]
    num = -1
    for it in range(iteration):
        group = {}
        for i in range(k):
            group[i] = []
        for j in x:
            tmp = []
            for i in range(k):
                tmp.append(euclidist(centers[i], j))
            group[np.argmin(tmp)].append(j.tolist())

        for i in range(k) :
            group_tmp = np.array(group[i])
            group_tmp = np.c_[group_tmp, np.full(len(group_tmp), i)]
            if i == 0  :
                grouped = group_tmp
            else :
                grouped = np.append(grouped, group_tmp, axis=0)

        centers_new = []
        for i in range(k):
            centers_new.append(gcenter(group[i]).tolist())
        centers_new = np.array(centers_new)
        if np.sum(centers - centers_new) == 0:
            break
        else :
            centers = centers_new
            log.append(grouped)
        num = num + 1
    return grouped, log, it

grouped, logs, it = cluster(coldatac1[1])

print(f'iter num:{it}', f'target : {grouped[43]}', f'id : {logs[2]}')

