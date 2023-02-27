import numpy as np
import pandas as pd


#10 predictive features 
# 
# SUIT  Ordinal:(1-4)
# Rank Numerical: (1-13)
# 1 output Goal: Ordinal (0-9)
"""
0: Nothing in hand; not a recognized poker hand 
      1: One pair; one pair of equal ranks within five cards
      2: Two pairs; two pairs of equal ranks within five cards
      3: Three of a kind; three equal ranks within five cards
      4: Straight; five cards, sequentially ranked with no gaps
      5: Flush; five cards with the same suit
      6: Full house; pair + different rank three of a kind
      7: Four of a kind; four equal ranks within five cards
      8: Straight flush; straight + flush
      9: Royal flush; {Ace, King, Queen, Jack, Ten} + flush
"""
#

def ReadDataset(path):
    features = np.array(['S1', 'R1','S2', 'R2','S3', 'R3','S4', 'R4','S5','R5','G'])
    columnsFeatures = pd.Series(features)
    trainingDataset = pd.read_csv(path,names=columnsFeatures)
    return trainingDataset

if __name__ == "__main__":
    trainingDataset = ReadDataset("./bin/resources/poker-hand-training-true.data")

    #There is no empty or nan values
    print("Checking null or NAN values...")
    print(trainingDataset.isnull().values.any())
    print("\nChecking numerical value for every columns ... i = integer")
    print([(c, trainingDataset[c].dtype.kind in 'i') for c in trainingDataset.columns])



    print("\nChecking numerical range predictive (1,13) columns")
    print([(c, trainingDataset[c].between(1,13).values.any()) for c in trainingDataset.columns.delete(-1)]) #removing goal

    print("\nChecking numerical range goal (0,9) column")
    print(trainingDataset["G"].between(0,9).values.any())


    #fare in modo che se c'è un false esce

    print("SIZE OF Training: (Records,Features)")
    print(trainingDataset.shape)
    
    print("Dropping duplicates...")
    trainingDataset = trainingDataset.drop_duplicates()
    print(trainingDataset.shape)

    print(trainingDataset)

    #We can regroup features: S1,C1 --->

    
