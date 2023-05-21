
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from math import sqrt,pi,exp
import seaborn as sns
from scipy.stats import zscore
import pickle
from sklearn.model_selection import train_test_split
from utility.UtilityFunctions import ReadDataset


from utility.UtilityFunctions import TreeBased,RandomForest,BayesComputingClassification


#TREE = "TREE"
#SVM_RBF = "SVM_RBF"
#BAYES = "BAYES"




#try it: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.multiclass







#Start main

if __name__ == "__main__":
    trainingDataset = ReadDataset("./bin/resources/poker-hand-training-true.data")
    testingDataset = ReadDataset("./bin/resources/poker-hand-testing.data")

    """
    with open('./bin/resources/training_encodedDf.pickle', 'rb') as data:
        trainingDataset_encoded = pickle.load(data)

    with open('./bin/resources/testing_encodedDf.pickle', 'rb') as data:
        testingDataset_encoded = pickle.load(data)

    with open('./bin/resources/training-sampled_encodedDf.pickle', 'rb') as data:
        trainingDataset_sampled_encoded = pickle.load(data)
    
    """
    
    with open('./bin/resources/training-sampled-dropped_encodedDf.pickle', 'rb') as data:
        training_sampled_encoded_dropped = pickle.load(data)

    with open('./bin/resources/test-dropped_encodedDf.pickle', 'rb') as data:
        test_encoded_dropped = pickle.load(data)


    print(training_sampled_encoded_dropped)
    print(test_encoded_dropped)

    #Our scope is classifying the poker hand by check card after card?!
    # 
    # ie. i see C1 as Ace Cuori
    # ie  i see C2 as Ace Rombo 
    # So result could Be poker or tris in which odd percentage?
    # 
    # 

    #Classification
    #LogReg as classification ?
    #Map  bayes
    #Tree
    #

    """
    y_train = trainingDataset['G']
    X_train = trainingDataset[['S1', 'R1','S2', 'R2','S3', 'R3','S4', 'R4','S5','R5']]

    y_test = testingDataset['G']
    X_test = testingDataset[['S1', 'R1','S2', 'R2','S3', 'R3','S4', 'R4','S5','R5']]


    X_train_encoded = trainingDataset_encoded.loc[:, trainingDataset_encoded.columns != 'label']
    y_train_encoded = trainingDataset_encoded.loc[:, trainingDataset_encoded.columns == 'label'].values.ravel()

    X_test_encoded = testingDataset_encoded.loc[:, testingDataset_encoded.columns != 'label']
    y_test_encoded = testingDataset_encoded.loc[:, testingDataset_encoded.columns == 'label'].values.ravel()

    X_train_sampled_encoded = trainingDataset_sampled_encoded.loc[:, trainingDataset_sampled_encoded.columns != 'label']
    y_train_sampled_encoded = trainingDataset_sampled_encoded.loc[:, trainingDataset_sampled_encoded.columns == 'label'].values.ravel()
    """
    #dropped
    X_train_sampled_encoded_dropped = training_sampled_encoded_dropped.loc[:, training_sampled_encoded_dropped.columns != 'label']
    y_train_sampled_encoded_dropped = training_sampled_encoded_dropped.loc[:, training_sampled_encoded_dropped.columns == 'label'].values.ravel()

    X_test_encoded_dropped = test_encoded_dropped.loc[:, test_encoded_dropped.columns != 'label']
    y_test_encoded_dropped = test_encoded_dropped.loc[:, test_encoded_dropped.columns == 'label'].values.ravel()


    print(X_train_sampled_encoded_dropped)
    print(y_train_sampled_encoded_dropped)

    TreeBased(X_train_sampled_encoded_dropped,y_train_sampled_encoded_dropped,X_test_encoded_dropped,y_test_encoded_dropped,1)
    RandomForest(X_train_sampled_encoded_dropped,y_train_sampled_encoded_dropped,X_test_encoded_dropped,y_test_encoded_dropped,1)
    BayesComputingClassification(X_train_sampled_encoded_dropped,y_train_sampled_encoded_dropped,X_test_encoded_dropped,y_test_encoded_dropped,1)

    exit()
    #TAKE a percentage
    X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = train_test_split(X_train_sampled_encoded, y_train_sampled_encoded, test_size=0.3, random_state=0, stratify=y_train_sampled_encoded)

    print("Taking percentage")
    print(X_train_encoded.shape)

    y_values = set(y_train.values)
    #print(y_values)
    
    """
    sns.distplot(trainingDataset['G'])
    plt.show()
    plt.close()
    """


    #print(trainingDataset.plot.area())

    #Density plot of our features
    #rankPlot = X_train[[ 'R1', 'R2', 'R3', 'R4','R5']]
    rankPlot = X_train_encoded[[ 'Asso', 'Due', 'Tre', 'Quattro', 'Cinque', 'Sei', 'Sette', 'Otto','Nove', 'Dieci', 'Principe','Regina','Re']]
    
    suitPlot = X_train[['S1','S2','S3','S4','S5']]

    rankPlot.plot.kde() 
    #plt.show() # gaussian partial
    plt.close()


    suitPlot.plot.kde()
    #plt.show()  #comb
    plt.close()


    #zScore#
    """
    X_train_scored = pd.DataFrame()
    X_test_scored = pd.DataFrame()
    for col in X_train.columns:
        X_train_scored[str(col)+'_zscore'] = zscore(X_train[col])
        X_test_scored[str(col)+'_zscore'] = zscore(X_test[col])


    rankPlot_scored = X_train_scored[[ 'R1_zscore', 'R2_zscore', 'R3_zscore', 'R4_zscore','R5_zscore']]
    rankPlot_scored.plot.kde() 
    #plt.show() # gaussian partial
    plt.close()


    suitPlot_scored = X_train_scored[['S1_zscore','S2_zscore','S3_zscore','S4_zscore','S5_zscore']]
    #suitPlot_scored.plot.kde()
    #plt.show()  #comb
    plt.close()

    print("Describe rank suit scored")
    print(rankPlot_scored.describe())
    print("\n\n===================\n\n")
    print(suitPlot_scored.describe())
    """


    #Transformation dummies with 1/0 of ranks => bernulli event?!


    activeEncoded=0
    #BayesComputingClassification(X_train,y_train,X_test,y_test,activeEncoded)
    #TreeBased(X_train,y_train,X_test,y_test,activeEncoded)
    #SvmBased(X_train,y_train,X_test,y_test,activeEncoded)


    print("ENCODED CLASSIFICATION")
    activeEncoded=1
    #TreeBased(X_train_encoded,y_train_encoded,X_test_encoded,y_test_encoded,activeEncoded,activeEncoded)
    #BayesComputingClassification(X_train_encoded,y_train_encoded,X_test_encoded,y_test_encoded,activeEncoded)
    #SvmBased(X_train_encoded,y_train_encoded,X_test_encoded,y_test_encoded,activeEncoded)



    









