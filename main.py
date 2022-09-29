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
coldata11 = []
coldata12 = []

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


cols = np.array(data1).T[0]
coldata1 = cols.tolist()
cols = np.array(data1).T[1]
coldata2 = cols.tolist()
cols = np.array(data1).T[2]
coldata3 = cols.tolist()
cols = np.array(data1).T[3]
coldata4 = cols.tolist()
cols = np.array(data1).T[4]
coldata5 = cols.tolist()
cols = np.array(data1).T[5]
coldata6 = cols.tolist()
cols = np.array(data1).T[6]
coldata7 = cols.tolist()
cols = np.array(data1).T[7]
coldata8 = cols.tolist()
cols = np.array(data1).T[8]
coldata9 = cols.tolist()
cols = np.array(data1).T[9]
coldata10 = cols.tolist()
cols = np.array(data1).T[10]
coldata11 = cols.tolist()
cols = np.array(data1).T[11]
coldata12 = cols.tolist()

for i in range(50) :
    print(i, coldata2[i])