# -*- coding: utf-8 -*-
import numpy as np
import os

def load_data():
    filename = "./datasets/data/mpg_data/auto-mpg.data"
    if not os.path.isfile(filename):
        import urllib.request
        dirname = os.path.dirname(filename)
        os.makedirs(dirname, exist_ok=True)
        urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data", filename)

    with open(filename, 'r') as f:
        rawtext = f.read()


    lines = rawtext.split('\n')
    N = len(lines) - 1
    M = 7
    X = np.zeros([N,M])
    Y = np.zeros([N,1])

    for i, line in enumerate(lines[:-1]):
        num, string = line.split('\t')
        num = num.split(' ')
        if "?" in num:
            continue
        num = [float(nu) for nu in num if nu != '']

        Y[i] = num[0]
        X[i,:] = num[1:]

    return X,Y
