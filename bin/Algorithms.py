
import numpy as np
import pandas as pd

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

#https://github.com/ss80226/MAP_estimation/tree/master/report
#https://python.quantecon.org/mle.html

def ReadDataset(path):
    features = np.array(['S1', 'R1','S2', 'R2','S3', 'R3','S4', 'R4','S5','R5','G'])
    columnsFeatures = pd.Series(features)
    trainingDataset = pd.read_csv(path,names=columnsFeatures)
    return trainingDataset

def MapComputing(training_df):
    count_list = [0, 0, 0]
    for data in training_df.values:
        count_list[int(data[0]-1)] += 1
    feature_1 = np.zeros(shape=[13, count_list[0]])
    feature_2 = np.zeros(shape=[13, count_list[1]])
    feature_3 = np.zeros(shape=[13, count_list[2]])

    for index, element in enumerate(training_df.values[0:count_list[0]]):
        for j in range(13):
            feature_1[j][index] = training_df.values[index][j+1]

    for index, element in enumerate(training_df.values[count_list[0]:count_list[0]+count_list[1]]):
        for j in range(13):
            feature_2[j][index] = training_df.values[count_list[0]+index][j+1]

    for index, element in enumerate(training_df.values[count_list[0]+count_list[1]:sum(count_list)]):
        for j in range(13):
            feature_3[j][index] = training_df.values[count_list[0]+count_list[1]+index][j+1]



if __name__ == "__main__":
    trainingDataset = ReadDataset("./bin/resources/poker-hand-training-true.data")

    #Our scope is classifying the poker hand by check card after card?!
    # 
    # ie. i see C1 as Ace Cuori
    # ie  i see C2 as Ace Rombo 
    # So result could Be poker or tris in which odd percentage?
    # 
    # 

    #Classification

    #LogReg as classification

    #Map  bayes

    y_train = trainingDataset['G']
    X_train = trainingDataset[['S1', 'R1','S2', 'R2','S3', 'R3','S4', 'R4','S5','R5']]

    MapComputing(X_train)










