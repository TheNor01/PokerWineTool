import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from statistics import mean
import itertools
import seaborn as sns

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
        return '%.2f'%(temp)
    else:
        temp = x/10
        return '%.1f'%(temp)

if __name__ == "__main__":
    trainingDataset = ReadDataset("./bin/resources/poker-hand-training-true.data")

    #How many rank and how many suit

    #approssimo all suo intero

    trainingSuit= trainingDataset[["S1", "S2", "S3", "S4", "S5","G"]]
    trainingRank = trainingDataset[["R1", "R2", "R3", "R4","R5","G"]]

    
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

    print("MOST COMMON IPOTETICAL CARD"+"("+str(meanSuit)+","+str(meanRank)+")")

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
    
    trainingCardsFull = trainingDataset[["C1", "C2", "C3", "C4", "C5","G"]]


    print(trainingCardsFull)
    trainingCards = trainingCardsFull[['C1', 'C2', 'C3','C4','C5']].applymap(DivideNumber)

    #Float or integer number as 4,2 -> 42, and then zscore all?

    trainingCards['G'] = trainingCardsFull['G']

    print("Mode")
    valuesMode = trainingCards[['C1', 'C2', 'C3','C4','C5']].astype(float).mode(axis=0).iloc[0]
    print(valuesMode)
    axis = valuesMode.plot(kind='bar')
    axis.set_title("Mode cards, First value is always the Suit")
    
    #plt.show()
    axis.clear()

    
    # Has it any sense?

    #b_plot = trainingCards.astype(float).boxplot(column = ["C1"]) 
    #b_plot.plot()
    #plt.show()


    #Which combination of poker hand is mostly frequent?
    auxCount = trainingCards[['C1', 'C2', 'C3','C4','C5']].value_counts()
    dictFreq = auxCount.to_dict() 
    print(dict(itertools.islice(dictFreq.items(), 5)))
    print(trainingCards[['C1', 'C2', 'C3','C4','C5']].value_counts(normalize=True).head(5))

    #Suit and Ranks correlation: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.corr.html

    
    #Using Pearson, maybe it's better to discard the case 0?

    
    #We want to see if a card id related to label
    print("Calculating CORR...")
    corr = trainingCards.astype(float).corrwith(trainingCards['G'],axis=0)
    print(corr[corr > 0.00001])
    
    #Seems that cards on C1,C3 are related to the poker hand, let's see which card are they
    filter = trainingCards["G"] != 0.0

    filterMax = trainingCards["G"] == 9.0
    #auxCount = trainingCards[['C3','G']].where(filter,inplace=True).head(5)
    #print(auxCount)

    #CARD 4,10 seems to be very common for G != 0
    print(trainingCards[['C3','G']].where(filter).value_counts(normalize=True).head(5))

    #CARD 4,9 seems to be very common for G != 0
    print(trainingCards[['C1','G']].where(filter).value_counts(normalize=True).head(5))

    #CARD 1,10 seems to be very common for G == 9
    print(trainingCards[['C1','G']].where(filterMax).value_counts(normalize=True).head(5))


    #Seems like there isn't a strong correlation between card and label
    #Also, correlation between features seems pretty rare, cause everytime there is a 


