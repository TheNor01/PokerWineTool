
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from math import sqrt,pi,exp
import seaborn as sns
from scipy.stats import zscore
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score
import pickle
from sklearn.metrics import classification_report
from utility.UtilityFunctions import plot_confusion_matrix,PlotTrainErrors


from utility.UtilityFunctions import ReadDataset

TREE = "TREE"
SVM_RBF = "SVM_RBF"

#https://github.com/ss80226/MAP_estimation/tree/master/report
#https://python.quantecon.org/mle.html


def BayesComputingClassification(X_train,y_train,X_test,y_test,activeEncoded):
    #https://scikit-learn.org/stable/modules/naive_bayes.html
    
    print("NAIVE BAYES CLASSIFICATION")
    clf = ComplementNB()
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    print("NAIVE BAYES Accuracy",accuracy_score(y_test, predictions))
    print("NAIVE BAYES Recall",accuracy_score(y_test, predictions))

    classes=np.unique(y_test)

    classesMetrics=['0','1','2','3','4','5','6','7','8','9']

    cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
    plt.close()

    plot_confusion_matrix(cm,classes,"BAYES",activeEncoded)

    print("BAYES")
    print(classification_report(y_test, predictions, target_names=classesMetrics))

    print("Plot train error bayes")
    PlotTrainErrors(X_train,y_train,clf,"BAYES",activeEncoded)

def TreeBased (X_train,y_train,X_test,y_test,activeEncoded):

    classesMetrics=['0','1','2','3','4','5','6','7','8','9']

    if(activeEncoded==1):
        print("Encoded tree active")

    print(TREE+": classification")
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train,y_train)

    predictions = clf.predict(X_test)
    print(TREE+":ACC",accuracy_score(y_test, predictions))
    print(TREE+":REC",recall_score(y_test, predictions))
    classes=np.unique(y_test)

    plt.close()
    cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
    plot_confusion_matrix(cm,classes,"TREE",activeEncoded)

    print("TREE Metric")
    print(classification_report(y_test, predictions, target_names=classesMetrics))
    PlotTrainErrors(X_train,y_train,clf,"TREE",activeEncoded)


def SvmBased(X_train,y_train,X_test,y_test,activeEncoded):
    classesMetrics=['0','1','2','3','4','5','6','7','8','9']
    #https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    clf_linear = SVC(kernel= 'linear', C=0.1,class_weight='balanced')
    clf_linear.fit(X_train, y_train)
    predictions = clf_linear.predict(X_test)
    print("SVM accuracy LINEAR",accuracy_score(y_test, predictions))
    classes=np.unique(y_test)

    plt.close()
    cm = confusion_matrix(y_test, predictions, labels=clf_linear.classes_)
    plot_confusion_matrix(cm,classes,"SVM_linear",activeEncoded)

    print("SVM linear")
    print(classification_report(y_test, predictions, target_names=classesMetrics))

    print("------------")

    clf_poly = SVC(kernel= 'poly', C=0.1,class_weight='balanced')
    clf_poly.fit(X_train, y_train)
    predictions = clf_poly.predict(X_test)
    print("SVM accuracy",accuracy_score(y_test, predictions))
    print("SVM recall",recall_score(y_test, predictions))
    classes=np.unique(y_test)

    plt.close()
    cm = confusion_matrix(y_test, predictions, labels=clf_poly.classes_)
    plot_confusion_matrix(cm,classes,"SVM_poly",activeEncoded)

    print("SVM poly")
    print(classification_report(y_test, predictions, target_names=classesMetrics))

    print("------------")

    clf_rbf = SVC(kernel= 'rbf', C=0.1,class_weight='balanced')
    clf_rbf.fit(X_train, y_train)
    predictions = clf_rbf.predict(X_test)
    print("SVM rbf accuracy",accuracy_score(y_test, predictions))
    print("SVM rfb recall",recall_score(y_test, predictions))
    classes=np.unique(y_test)

    plt.close()
    cm = confusion_matrix(y_test, predictions, labels=clf_rbf.classes_)
    plot_confusion_matrix(cm,classes,"SVM_rbf",activeEncoded)

    print("SVM RBF")
    print(classification_report(y_test, predictions, target_names=classesMetrics))
    PlotTrainErrors(X_train,y_train,clf_rbf,"SVM RBF",activeEncoded)



#Start main

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
    y_train_encoded = trainingDataset_encoded.loc[:, trainingDataset_encoded.columns == 'label'].values.ravel()


    X_test_encoded = testingDataset_encoded.loc[:, testingDataset_encoded.columns != 'label']
    y_test_encoded = testingDataset_encoded.loc[:, testingDataset_encoded.columns == 'label'].values.ravel()


    y_values = set(y_train.values)
    #print(y_values)
    
    # for value in y_values:
    #      tmp = X_train[X_train.G == value]
    #      ax = tmp.hist(column=['S1', 'R1','S2', 'R2','S3', 'R3','S4', 'R4','S5','R5'])
    #      plt.title("Histogram of :"+str(value))  #Fix doesn't show
    #      #plt.show()
    #      plt.cla()
    """
    sns.distplot(trainingDataset['G'])
    plt.show()
    plt.close()
    """


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


    #Transformation dummies with 1/0 of ranks => bernulli event?!


    print("Describe rank suit scored")
    print(rankPlot_scored.describe())
    print("\n\n===================\n\n")
    print(suitPlot_scored.describe())

    activeEncoded=0
    #BayesComputingClassification(X_train,y_train,X_test,y_test)
    print("\n\n========\n\n")
    #negative 
    #BayesComputingClassification(X_train_scored,y_train,X_test,y_test)
    print("\n\n===================\n\n")
    #TreeBased(X_train,y_train,X_test,y_test)
    print("\n\n========\n\n")
    TreeBased(X_train_scored,y_train,X_test_scored,y_test,activeEncoded)


    print("ENCODED CLASSIFICATION")
    activeEncoded=1
    TreeBased(X_train_encoded,y_train_encoded,X_test_encoded,y_test_encoded,activeEncoded)
    #BayesComputingClassification(X_train_encoded,y_train_encoded,X_test_encoded,y_test_encoded)
    #SvmBased(X_train_encoded,y_train_encoded,X_test_encoded,y_test_encoded)



    









