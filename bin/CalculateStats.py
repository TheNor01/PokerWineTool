import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from statistics import mean
import itertools

#10 predictive features 
# 
# SUIT  Ordinal:(1-4) {Hearts, Spades, Diamonds, Clubs} //picche
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
#  https://www.chemical-ecology.net/java/possible.htm#:~:text=The%20total%20number%20of%20possible,drawn%20(answer%202%2C598%2C960%20combinations).

def ReadDataset(path):
    features = np.array(['S1', 'R1','S2', 'R2','S3', 'R3','S4', 'R4','S5','R5','G'])
    columnsFeatures = pd.Series(features)
    trainingDataset = pd.read_csv(path,names=columnsFeatures)
    return trainingDataset

def DivideNumber(x):
    if(x>100): 
        temp = x/100
        return round(temp, 2)
    else:
        temp = x/10
        return round(temp, 1)

if __name__ == "__main__":
    trainingDataset = ReadDataset("./bin/resources/poker-hand-training-true.data")

    #How many rank and how many suit

    #approssimo all suo intero

    trainingSuit= trainingDataset[["S1", "S2", "S3", "S4", "S5"]]
    trainingRank = trainingDataset[["R1", "R2", "R3", "R4","R5"]]

    
    print(trainingDataset)

    #print(trainingSuit.describe().astype(int))
    print("\n---")
    #print(trainingRank.describe().astype(int))


    axis = trainingSuit.mean().plot(kind='bar')
    axis.set_title("Avegerage Suit card")
    axis.set_xlabel("Suit values")
    axis.set_ylabel("Mean")


    #plt.show()
    meanSuit  = mean(trainingSuit.mean(axis=0).values).astype(int)
    print("AVERAGE SUIT IS:",meanSuit)

    axis = trainingRank.mean().plot(kind='bar')
    axis.set_title("Avegerage Rank card")
    axis.set_xlabel("Rank values")
    axis.set_ylabel("Mean")

    #plt.show()
    axis.clear()

    meanRank = mean(trainingRank.mean(axis=0).values).astype(int)
    print("AVERAGE RANK IS:",meanRank)

    print("MOST COMMON CARD"+"("+str(meanSuit)+","+str(meanRank)+")")

    #What if we calculate mean by grouping cards?

    #trainingDataset['C1'] = '('+trainingSuit['S1'].map(str) + ',' + trainingRank['R1'].map(str)+')'
    #trainingDataset['C2'] = '('+trainingSuit['S2'].map(str) + ',' + trainingRank['R2'].map(str)+')'
    #trainingDataset['C3'] = '('+trainingSuit['S3'].map(str) + ',' + trainingRank['R3'].map(str)+')'
    #trainingDataset['C4'] = '('+trainingSuit['S4'].map(str) + ',' + trainingRank['R4'].map(str)+')'
    #trainingDataset['C5'] = '('+trainingSuit['S5'].map(str) + ',' + trainingRank['R5'].map(str)+')'

    trainingDataset['C1'] = (trainingSuit['S1'].map(str)+trainingRank['R1'].map(str)).astype(int)
    trainingDataset['C2'] = (trainingSuit['S2'].map(str)+trainingRank['R2'].map(str)).astype(int)
    trainingDataset['C3'] = (trainingSuit['S3'].map(str)+trainingRank['R3'].map(str)).astype(int)
    trainingDataset['C4'] = (trainingSuit['S4'].map(str)+trainingRank['R4'].map(str)).astype(int)
    trainingDataset['C5'] = (trainingSuit['S5'].map(str)+trainingRank['R5'].map(str)).astype(int)
    
    trainingCards = trainingDataset[["C1", "C2", "C3", "C4", "C5"]]

    

    trainingCards = trainingCards.applymap(DivideNumber) #to be fixed


    print(trainingCards)
    
    valuesMode = trainingCards.mode(axis=0).iloc[0]
    print(valuesMode)
    axis = valuesMode.plot(kind='bar')
    axis.set_title("Mode cards, First value is always the Suit")
    
    #plt.show()
    axis.clear()

    b_plot = trainingCards.boxplot(column = ['C1', 'C2', 'C3','C4','C5']) 
    b_plot.plot()
    #plt.show()


    #Which combination of cards is mostly frequent?
    auxCount = trainingCards[['C1', 'C2', 'C3','C4','C5']].value_counts()
    dictFreq = auxCount.to_dict() 
    print(dict(itertools.islice(dictFreq.items(), 5)))
    print(trainingCards[['C1', 'C2', 'C3','C4','C5']].value_counts(normalize=True).head(5))


    #Which combination of cards is relevant for a Goal?



