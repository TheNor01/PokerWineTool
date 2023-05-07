import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from sklearn.tree import DecisionTreeClassifier

"""
POKER has 52 card, 13 carte per seme.

Un rank può avere 4 semi

Abbiamo un dataset con un training set di 

"""


with open('./bin/resources/training-sampled-dropped_encodedDf.pickle', 'rb') as data:
        droppedTR_encoded = pickle.load(data)

print(droppedTR_encoded)