import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
from collections import Counter
from numpy.random import seed
from numpy.random import randint
import random
import pickle
from imblearn import over_sampling
import tkinter

from utility.UtilityFunctions import ReadDataset,ApplyTrasformation


#10 predictive features 
# 
# SUIT  Ordinal:(1-4) representing {Hearts, Spades, Diamonds, Clubs} //picche
# Rank Numerical: (1-13) (Ace, 2, 3, ... , Queen, King)
# 1 output Goal: Ordinal (0-9)
"""
      0: Nothing in hand; not a recognized poker hand 
      1: One pair; one pair of equal ranks within five cards // coppia
      2: Two pairs; two pairs of equal ranks within five cards // doppia coppia
      3: Three of a kind; three equal ranks within five cards //tris
      4: Straight; five cards, sequentially ranked with no gaps //scala
      5: Flush; five cards with the same suit //colore
      6: Full house; pair + different rank three of a kind // Full (tris + coppia)
      7: Four of a kind; four equal ranks within five cards (poker)
      8: Straight flush; straight + flush // scala colore
      9: Royal flush; {Ace, King, Queen, Jack, Ten} + flush // scala reale

      suit_name = {
        1: 'Cuori',
        2: 'Picche',
        3: 'Quadri',
        4: 'Fiori'
        }
"""

#Questions 
    #We can regroup features: S1,C1 --->
    #We can balance samples??? For instance adding more examples of less frequent classes?

    #What about discard 0 label rows, if a classification has got low prob, it is a 0 label, unclassified

    #Order Column ascendent based on RANK?
#




def CheckIntegrityDataset(dataset):
    #Method used to detect strange values, according to problem specification

    print("Checking null or NAN values...")

    checkNan = dataset.isnull().values.any() #returns true if there is one true at least
    print(checkNan)

    print("\nChecking numerical value for every columns ... i = integer")
    checksNumerical = [(c, dataset[c].astype(str).str.isdigit().values.any()) for c in dataset.columns]
    print(checksNumerical)

    print("\nChecking range predictive (1,13) columns")
    checkRange1 = [(c, trainingDataset[c].between(1,13).values.any()) for c in trainingDataset.columns.delete(-1)]
    print(checkRange1) #removing goal

    print("\nChecking range goal (0,9) column")
    checkRange2 = trainingDataset["G"].between(0,9).values.any()
    print(checkRange2)



def PrintShapeGraph(dataset):
    print("SIZE OF : (Records,Features)")
    print(dataset.shape)

    g_classes = len(set(dataset['G'].values))  # count distinct values label
    poker_hands = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    #poker_hands = [1, 2, 3, 4, 5, 6, 7, 8, 9]


    print(g_classes)
    cls = {}
    for i in range(g_classes):
        cls[i] = len(dataset[dataset.G==i])
    print(cls)

    #classes are unbalanced

    #Plot histogram

    plt.bar(poker_hands, [cls[i] for i in poker_hands], align='center')
    plt.xlabel('classes id')
    plt.ylabel('Number of instances')
    plt.title("Classes of Dataset")
    plt.show()

def PlotCrossTab(trainingDataset):

    #We plot a heatmap starting from all values rank and suit, based on k random index
    #we obtain the most common card

    allS = []
    allR = []
    for col in trainingDataset.columns:
        #print(col)
        filteredCol = list(trainingDataset[col].values)
        if(str(col).startswith("S")):
            print("S",str(col))
            allS=allS+filteredCol
        elif(str(col).startswith("R")):
            print("R",str(col))
            allR=allR+filteredCol
        else:
            continue

    s = allS
    r = allR

    random.seed(10)
    #Incremental k sample
    # generate some integersm, random


    kSample=4000
    indexes = random.sample(range(1, len(allS)), k=kSample)
    #print(indexes)
    subS  = [s[i] for i in indexes]
    subR  = [r[i] for i in indexes]

    #I don't make any difference but a R5 could be a R1 in terms of position
    dfHeat = pd.DataFrame()
    dfHeat["subS"] = subS
    dfHeat["subR"] = subR
    df2 = pd.crosstab(dfHeat['subS'], dfHeat['subR']).div(len(dfHeat))
    sns.heatmap(df2, annot=True)

    plt.xlabel('Ranks', fontsize=14)
    plt.ylabel('Suits', fontsize=14)
    plt.title("Cards occurrence")
    #plt.close()
    plt.show()



def OversampleDataset(dataset,target):

    y_train = dataset['G']
    X_train = dataset.loc[:, dataset.columns != 'G']

    print("OVERSAMPLING")
    
    #oversample = over_sampling.RandomOverSampler(sampling_strategy="minority")
    dictSamples= { 2: target, 3: target, 4: target, 5: target, 6 : target, 7: target, 8: target, 9: target}
    oversample = over_sampling.RandomOverSampler(sampling_strategy=dictSamples)
    
    X_over, y_over = oversample.fit_resample(X_train, y_train)

    print(X_over.shape)

    mergedDf = pd.concat([X_over, y_over], axis=1)
    #print(mergedDf)

    return mergedDf

def swapFirst(x):
    return 'R'+x[1:]

def CreatePartialTR(trainingDataset):

    numberDrops = 2

    dropIndex = ['S1','S2','S3','S4','S5']

    Allrows = []
    for index,rows in trainingDataset.iterrows():
        #print(rows)
        #print(type(rows))
        sample_index_toDrop = random.sample(dropIndex, numberDrops)
        #print("DELETING SUIT",sample_index_toDrop)

        rank_to_delete= list(map(swapFirst, sample_index_toDrop))
        #print("DELETING RANKS",rank_to_delete)

        sample_index_toDrop.extend(rank_to_delete)
        #print(sample_index_toDrop)
        newRow = rows.drop(labels=sample_index_toDrop).reset_index(drop = True).tolist()
        #print(newRow)
        if(len(newRow)) > 7: 
            print("WARNING")
            print(sample_index_toDrop)


        Allrows.append(newRow)

    newColumns = ['S1', 'R1', 'S2', 'R2', 'S3', 'R3', 'G']
    droppedTR = pd.DataFrame(Allrows, columns=newColumns)

    print(droppedTR)
    print(droppedTR.shape)

    with open("./bin/resources/droppedTR.pickle", 'wb') as output:
            pickle.dump(droppedTR, output)
    
    return droppedTR
            
#Main start

if __name__ == "__main__":
    #fare in modo che se c'è un false esce
    print("### TRAINING ###")
    trainingDataset = ReadDataset("./bin/resources/poker-hand-training-true.data")
    testingDataset = ReadDataset("./bin/resources/poker-hand-testing.data")

    print(trainingDataset)

    print("DROPPING...")
    partialTR = CreatePartialTR(trainingDataset)
    partialTS = CreatePartialTR(testingDataset)


    # define oversampling strategy
    target = 3000
    oversampledDf_dropped = OversampleDataset(partialTR,target)

    print("ENCODING TRAINING sampled dropped")
    training_encoded_sampled_Df_dropped = ApplyTrasformation(oversampledDf_dropped,"training-sampled-dropped")
    test_encoded_sampled_Df_dropped = ApplyTrasformation(partialTS,"test-dropped")

    exit()

    CheckIntegrityDataset(trainingDataset)

    print(trainingDataset)

    print("Dropping duplicates...")
    trainingDataset = trainingDataset.drop_duplicates()


    #Info shape training
    PrintShapeGraph(trainingDataset)


    #Loading testing
    print("\n\n### TESTING ###")
    testingDataset = ReadDataset("./bin/resources/poker-hand-testing.data")
    CheckIntegrityDataset(testingDataset)

    plt.close()
    PrintShapeGraph(testingDataset)


    plt.close()
    PlotCrossTab(trainingDataset)

    ...
   


    plt.close()
    PrintShapeGraph(oversampledDf)



    #Linear transformation from 11D to 18D


    #We want to obtain a trasformation regardless card position and order.
    # Basically the main idea is to count how many card there are in each valuation
    # With rank and suit. Thanks to it, we have a standard rapresentation of training/test set

    print("ENCODING TRAINING")
    training_encodedDf = ApplyTrasformation(trainingDataset,"training")

    print("ENCODING TRAINING sampled")
    training_encoded_sampled_Df = ApplyTrasformation(oversampledDf,"training-sampled")

    print("ENCODING TEST")
    testing_encodedDf = ApplyTrasformation(testingDataset,"testing")

    
    
    

