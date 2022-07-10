# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def load_data_file(filename):
    data = pd.read_csv(filename)
    T = np.array(data['Temperature']).reshape([-1,1])
    # convert to appropriate scale
    #X = 10**6/(T**2)
    X = T
    Y = np.array(data['D47']).reshape([-1,1])
    return X,Y

def load_data():
    Xtr, Ytr = load_data_file("./datasets/data/isotope/Training.DS3.csv")
    Xte, Yte = load_data_file("./datasets/data/isotope/Test.DS3.csv")
    return Xtr, Xte, Ytr, Yte
