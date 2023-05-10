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


Rispiego:
Ad una mano completa di carte (5) è possibile assegnare un punteggio di valore (esistono sistemi standard di punteggio).
Una spiegazione lunga ma semplice la trovi forse qui: http://nsayer.blogspot.com/2007/07/algorithm-for-evaluating-poker-hands.html


Quel che suggerisco è valutare le 5-tuple di carte nei dati originali e assegnare un punteggio.
In base ad una soglia "etichettare" i dati originali Winning-NoWinning.
Eliminare random 2 carte da ogni cinquina e mantenere l'etichetta Winning-NoWinning assegnata prima.
Su questo nuovo data set cercare di fare classificazione usando le due classi Winning-noWinning


Si tratta solo di "CLASSIFICARE" non di metter ein piedi un algoritmo che date tre carte stimi la probabilità di vincita o men. Questo sarebbe troppo complicato e richiede combinatorica piuttosto che ML.
Spero sia utile

"""


with open('./bin/resources/training-sampled-dropped_encodedDf.pickle', 'rb') as data:
        droppedTR_encoded = pickle.load(data)

print(droppedTR_encoded)