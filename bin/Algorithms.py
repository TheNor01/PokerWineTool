
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from math import sqrt,pi,exp
import seaborn as sns
from scipy.stats import zscore
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
import pickle


from utility.UtilityFunctions import plot_confusion_matrix

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

hand_name = {
    0: 'Null',
    1: 'Coppia',
    2: 'Doppia Coppia',
    3: 'Tris',
    4: 'Scala',
    5: 'Colore',
    6: 'FULL',
    7: 'Poker',
    8: 'Scala colore',
    9: 'Scala Reale',
    }

suit_name = {
    1: 'Cuori',
    2: 'Picche',
    3: 'Quadri',
    4: 'Fiori'
}



#https://github.com/ss80226/MAP_estimation/tree/master/report
#https://python.quantecon.org/mle.html

def ReadDataset(path):
    features = np.array(['S1', 'R1','S2', 'R2','S3', 'R3','S4', 'R4','S5','R5','G'])
    columnsFeatures = pd.Series(features)
    trainingDataset = pd.read_csv(path,names=columnsFeatures)
    return trainingDataset


def BayesComputingClassification(X_train,y_train,X_test,y_test):
    #https://scikit-learn.org/stable/modules/naive_bayes.html
    
    print("NAIVE BAYES CLASSIFICATION")

    clf = ComplementNB()
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    print("NAIVE BAYES accuracy",accuracy_score(y_test, predictions))

    plt.close()
    classes=np.unique(y_test)
    cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
    plot_confusion_matrix(cm,classes,"BAYES")

def TreeBased (X_train,y_train,X_test,y_test):
    print("TREE CLASSIFICATION")
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train,y_train)

    predictions = clf.predict(X_test)
    print("TREE accuracy",accuracy_score(y_test, predictions))
    classes=np.unique(y_test)

    plt.close()
    cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
    plot_confusion_matrix(cm,classes,"TREE")



def SvmBased(X_train,y_train,X_test,y_test):

    #https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    clf = SVC(kernel= 'poly', random_state=1, C=0.1,class_weight='balanced')
    clf.fit(X_train, y_train)


    predictions = clf.predict(X_test)
    print("SVM accuracy",accuracy_score(y_test, predictions))
    classes=np.unique(y_test)

    plt.close()
    cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
    plot_confusion_matrix(cm,classes,"SVM")



if __name__ == "__main__":
    trainingDataset = ReadDataset("./bin/resources/poker-hand-training-true.data")
    testingDataset = ReadDataset("./bin/resources/poker-hand-testing.data")

    with open('./bin/resources/training_encodedDf.pickle', 'rb') as data:
        trainingDataset_encoded = pickle.load(data)

    with open('./bin/resources/testing_encodedDf.pickle', 'rb') as data:
        testingDataset_encoded = pickle.load(data)


    print(trainingDataset_encoded.shape)


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

    y_test = testingDataset['G']
    X_test = testingDataset[['S1', 'R1','S2', 'R2','S3', 'R3','S4', 'R4','S5','R5']]


    X_train_encoded = trainingDataset_encoded.loc[:, trainingDataset_encoded.columns != 'label']
    y_train_encoded = trainingDataset_encoded.loc[:, trainingDataset_encoded.columns == 'label']


    X_test_encoded = testingDataset_encoded.loc[:, testingDataset_encoded.columns != 'label']
    y_test_encoded = testingDataset_encoded.loc[:, testingDataset_encoded.columns == 'label']


    

    y_values = set(y_train.values)
    print(y_values)
    
    # for value in y_values:
    #      tmp = X_train[X_train.G == value]
    #      ax = tmp.hist(column=['S1', 'R1','S2', 'R2','S3', 'R3','S4', 'R4','S5','R5'])
    #      plt.title("Histogram of :"+str(value))  #Fix doesn't show
    #      #plt.show()
    #      plt.cla()

    sns.distplot(trainingDataset['G'])
    
    #plt.show()
    plt.close()

    #print(trainingDataset.plot.area())

    #Density plot of our features
    rankPlot = X_train[[ 'R1', 'R2', 'R3', 'R4','R5']]
    suitPlot = X_train[['S1','S2','S3','S4','S5']]

    rankPlot.plot.kde() 
    #plt.show() # gaussian partial
    plt.close()

    suitPlot.plot.kde()
    #plt.show()  #comb
    plt.close()


    
    X_train_scored = pd.DataFrame()
    for col in X_train.columns:
        X_train_scored[str(col)+'_zscore'] = zscore(X_train[col])


    rankPlot_scored = X_train_scored[[ 'R1_zscore', 'R2_zscore', 'R3_zscore', 'R4_zscore','R5_zscore']]
    rankPlot_scored.plot.kde() 
    #plt.show() # gaussian partial
    plt.close()


    suitPlot_scored = X_train_scored[['S1_zscore','S2_zscore','S3_zscore','S4_zscore','S5_zscore']]
    suitPlot_scored.plot.kde()
    #plt.show()  #comb
    plt.close()

    #Transformation dummies with 1/0 of ranks => bernulli event?!

    print(rankPlot_scored.describe())

    print("\n\n===================\n\n")
    print(suitPlot_scored.describe())

    BayesComputingClassification(X_train,y_train,X_test,y_test)
    print("\n\n========\n\n")
    #BayesComputingClassification(X_train_scored,y_train,X_test,y_test)
    print("\n\n===================\n\n")
    TreeBased(X_train,y_train,X_test,y_test)
    print("\n\n========\n\n")
    TreeBased(X_train_scored,y_train,X_test,y_test)


    print("ENCODED CLASSIFICATION")

    TreeBased(X_train_encoded,y_train_encoded,X_test_encoded,y_test_encoded)
    BayesComputingClassification(X_train_encoded,y_train_encoded,X_test_encoded,y_test_encoded)
    SvmBased(X_train_encoded,y_train_encoded,X_test_encoded,y_test_encoded)




   










