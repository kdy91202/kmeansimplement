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
num = -1

with open('assignment2_input.txt', 'r') as file:
    for line in file:
        data = line.strip().split('\t')
        data = line.strip().split('\n')

        if num < 49:
            print(num, line)
        if num >= 49 and num < 98:
            data2.append(line)
            print(num, data2)

        num = num + 1
"""     if line > '99' and line <= '149':
            data3.append(float(data3[data]))
        if line > '149' and line <= '199':
            data4.append(float(data4[data]))
        if line > '199' and line <= '249':
            data5.append(float(data5[data]))
        if line > '249' and line <= '299':
            data6.append(float(data6[data]))
        if line > '299' and line <= '349':
            data7.append(float(data7[data]))
        if line > '349' and line <= '399':
            data8.append(float(data8[data]))
        if line > '399' and line <= '449':
            data9.append(float(data9[data]))
        if line > '449' and line <= '499':
            data10.append(float(data10[data])) """