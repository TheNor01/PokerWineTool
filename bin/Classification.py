
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from math import sqrt,pi,exp
import seaborn as sns
from scipy.stats import zscore
from sklearn.naive_bayes import ComplementNB,MultinomialNB
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utility.UtilityFunctions import plot_confusion_matrix,PlotTrainErrors


from utility.UtilityFunctions import ReadDataset

TREE = "TREE"
SVM_RBF = "SVM_RBF"
BAYES = "BAYES"

classesMetrics=['0','1','2','3','4','5','6','7','8','9']


#try it: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.multiclass



#bayes classification
def BayesComputingClassification(X_train,y_train,X_test,y_test,activeEncoded):


    #https://scikit-learn.org/stable/modules/naive_bayes.html
    print("\n\n========\n\n")
    print(BAYES+" CLASSIFICATION")
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    print("NAIVE BAYES Complement Accuracy",accuracy_score(y_test, predictions))

    classes=np.unique(y_test)
    cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
    plt.close()

    plot_confusion_matrix(cm,classes,BAYES,activeEncoded)

    print("BAYES report classification")
    print(classification_report(y_test, predictions, target_names=classesMetrics,zero_division=1))

    print("Plotting train error bayes....")
    PlotTrainErrors(X_train,y_train,clf,BAYES,activeEncoded)

############################################################################

#tree classification
def TreeBased (X_train,y_train,X_test,y_test,activeEncoded):
    print("\n\n========\n\n")
    print(TREE+": classification")

    #try it https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    clf = DecisionTreeClassifier(criterion='gini',max_depth=6,class_weight='balanced')
    clf = clf.fit(X_train,y_train)

    predictions = clf.predict(X_test)
    print(TREE+":ACC",accuracy_score(y_test, predictions))
    classes=np.unique(y_test)

    plt.close()
    cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
    plot_confusion_matrix(cm,classes,"TREE",activeEncoded)

    print("TREE Metric")
    print(classification_report(y_test, predictions, target_names=classesMetrics))
    PlotTrainErrors(X_train,y_train,clf,"TREE",activeEncoded)

    #Plotting tree
    plt.figure(figsize=(12,12))
    plot_tree(clf, fontsize=6)
    plt.savefig('./Images/treePlot.png', dpi=100)






#######################################################################################
#svm classification
def SvmBased(X_train,y_train,X_test,y_test,activeEncoded):
    print("\n\n========\n\n")
    print("SVM classification linear")
    #https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

    clf_linear = SVC(kernel= 'linear', C=0.1,class_weight='balanced')
    clf_linear.fit(X_train, y_train)

    predictions = clf_linear.predict(X_test)
    print("SVM accuracy LINEAR",accuracy_score(y_test, predictions))
    classes=np.unique(y_test)

    plt.close()
    cm = confusion_matrix(y_test, predictions, labels=clf_linear.classes_)
    plot_confusion_matrix(cm,classes,"SVM_linear",activeEncoded)

    print("SVM linear REPORT")
    print(classification_report(y_test, predictions, target_names=classesMetrics))

    print("------------")
    #----------------------------------------------------------------------------------------
    print("SVM classification poly")
    clf_poly = SVC(kernel= 'poly', C=0.1,class_weight='balanced')
    clf_poly.fit(X_train, y_train)
    predictions = clf_poly.predict(X_test)
    print("SVM POLY accuracy",accuracy_score(y_test, predictions))
    classes=np.unique(y_test)

    plt.close()
    cm = confusion_matrix(y_test, predictions, labels=clf_poly.classes_)
    plot_confusion_matrix(cm,classes,"SVM_poly",activeEncoded)

    print("SVM poly Report")
    print(classification_report(y_test, predictions, target_names=classesMetrics))

    print("------------")
    #----------------------------------------------------------------------------------------
    print("SVM classification rbf")
    clf_rbf = SVC(kernel= 'rbf', C=0.1,class_weight='balanced')
    clf_rbf.fit(X_train, y_train)
    predictions = clf_rbf.predict(X_test)
    
    print("SVM rbf accuracy",accuracy_score(y_test, predictions))
    classes=np.unique(y_test)

    plt.close()
    cm = confusion_matrix(y_test, predictions, labels=clf_rbf.classes_)
    plot_confusion_matrix(cm,classes,"SVM_rbf",activeEncoded)

    print("SVM RBF Report")
    print(classification_report(y_test, predictions, target_names=classesMetrics))
    #PlotTrainErrors(X_train,y_train,clf_rbf,"SVM RBF",activeEncoded)

    print("Class weights")
    print(clf_rbf.class_weight_)



#Start main

if __name__ == "__main__":
    trainingDataset = ReadDataset("./bin/resources/poker-hand-training-true.data")
    testingDataset = ReadDataset("./bin/resources/poker-hand-testing.data")

    with open('./bin/resources/training_encodedDf.pickle', 'rb') as data:
        trainingDataset_encoded = pickle.load(data)

    with open('./bin/resources/testing_encodedDf.pickle', 'rb') as data:
        testingDataset_encoded = pickle.load(data)

    with open('./bin/resources/training-sampled_encodedDf.pickle', 'rb') as data:
        trainingDataset_sampled_encoded = pickle.load(data)

    print(trainingDataset_encoded.shape)
    print(trainingDataset_sampled_encoded.shape)


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
    SvmBased(X_train_encoded,y_train_encoded,X_test_encoded,y_test_encoded,activeEncoded)



    









