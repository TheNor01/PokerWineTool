import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.decomposition import PCA
import matplotlib.cm as cm



from bin.utility.UtilityFunctions import TreeBased,RandomForest,BayesComputingClassification,SvmBased


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
def PrintShapeGraph(dataset):
    print("SIZE OF : (Records,Features)")
    print(dataset.shape)

    g_classes = len(set(dataset['isWinning'].values))  # count distinct values label
    poker_hands = list(set(dataset['isWinning'].values))

    print(g_classes)
    cls = {}
    for i in range(g_classes):
        cls[i] = len(dataset[dataset.isWinning==i])
    print(cls)

    #classes are unbalanced

    #Plot histogram

    plt.bar(poker_hands, [cls[i] for i in poker_hands], align='center')
    plt.xlabel('classes id')
    plt.ylabel('Number of instances')
    plt.title("Classes of Dataset")
    plt.show()

with open('./bin/resources/training-dropped_encodedDf.pickle', 'rb') as data:
        droppedTR_encoded = pickle.load(data)

with open('./bin/resources/testing-dropped_encodedDf.pickle', 'rb') as data:
        droppedTS_encoded = pickle.load(data)



print(droppedTR_encoded)

droppedTR_encoded = droppedTR_encoded.drop('label', axis=1)
droppedTS_encoded = droppedTS_encoded.drop('label', axis=1)

X_train_encoded = droppedTR_encoded.loc[:, droppedTR_encoded.columns != 'isWinning']
y_train_encoded = droppedTR_encoded.loc[:, droppedTR_encoded.columns == 'isWinning'].values.ravel()

X_test_encoded = droppedTS_encoded.loc[:, droppedTS_encoded.columns != 'isWinning']
y_test_encoded = droppedTS_encoded.loc[:, droppedTS_encoded.columns == 'isWinning'].values.ravel()


PrintShapeGraph(droppedTR_encoded)
PrintShapeGraph(droppedTS_encoded)


print(X_train_encoded)
print(X_test_encoded)

TreeBased(X_train_encoded,y_train_encoded,X_test_encoded,y_test_encoded,1)
RandomForest(X_train_encoded,y_train_encoded,X_test_encoded,y_test_encoded,1)


BayesComputingClassification(X_train_encoded,y_train_encoded,X_test_encoded,y_test_encoded,1)
SvmBased(X_train_encoded,y_train_encoded,X_test_encoded,y_test_encoded,1)



#sample