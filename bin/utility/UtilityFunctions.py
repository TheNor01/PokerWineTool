from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import itertools
from sklearn.model_selection import ShuffleSplit,LearningCurveDisplay
import pickle
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.naive_bayes import ComplementNB,MultinomialNB

#classesMetrics=['0','1','2','3','4','5','6','7','8','9']
#classesMetrics=['0','1']

TREE = "TREE"
SVM_RBF = "SVM_RBF"
BAYES = "BAYES"

def plot_confusion_matrix(cm, classes, classifier,encodedActive,normalize=False, cmap=cm.Blues, png_output="./Images", show=True):
       
        title='Confusion matrix OF '+classifier+ "- ENC:"+str(encodedActive)
        # Calculate chart area size
        leftmargin = 0.5 # inches
        rightmargin = 0.5 # inches
        categorysize = 0.5 # inches
        figwidth = leftmargin + rightmargin + (len(classes) * categorysize)           

        f = plt.figure(figsize=(figwidth, figwidth))

        # Create an axes instance and ajust the subplot size
        ax = f.add_subplot(111)
        ax.set_aspect(1)
        f.subplots_adjust(left=leftmargin/figwidth, right=1-rightmargin/figwidth, top=0.94, bottom=0.1)

        res = ax.imshow(cm, interpolation='nearest', cmap=cmap)

        plt.title(title)
        plt.colorbar(res)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_yticklabels(classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        # plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        if png_output is not None:
            os.makedirs(png_output, exist_ok=True)
            if(encodedActive):
                f.savefig(os.path.join(png_output,'Encoded confusion_matrix_'+classifier+'.png'), bbox_inches='tight')
            else:
                f.savefig(os.path.join(png_output,'confusion_matrix_'+classifier+'.png'), bbox_inches='tight')

        if show:
            plt.show()
            plt.close(f)
        else:
            plt.close(f)

def PlotTrainErrors(X_train,y_train,classifier,clfAsString,encodedActive):


    plt.close()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

    common_params = {
        "X": X_train,
        "y": y_train,
        "train_sizes": np.linspace(0.1, 1.0, 5),
        "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
        "score_type": "both",
        "n_jobs": 4,
        "line_kw": {"marker": "o"},
        "std_display_style": "fill_between",
        "score_name": "Accuracy",
    }

    title = f"Learning Curve for {classifier.__class__.__name__} ENC {encodedActive}"
    for ax_idx, estimator in enumerate([classifier]):
        LearningCurveDisplay.from_estimator(estimator, **common_params, ax=ax[ax_idx])
        handles, label = ax[ax_idx].get_legend_handles_labels()
        ax[ax_idx].legend(handles[:2], ["Training Score", "Test Score"])
        ax[ax_idx].set_title(title)
    
    plt.show()

    if(encodedActive):
        fig.savefig('./Images/Encoded_trainError_'+clfAsString+'.png')
    else:
        fig.savefig('./Images/trainError_'+clfAsString+'.png')

    #fig.show()


def ReadDataset(path):


    features = np.array(['S1', 'R1','S2', 'R2','S3', 'R3','S4', 'R4','S5','R5','G'])
    columnsFeatures = pd.Series(features)
    trainingDataset = pd.read_csv(path,names=columnsFeatures)

    # making boolean series for a team name
    #filter0 = trainingDataset["G"] != 0
  
    # filtering data
    #trainingDataset.where(filter0, inplace = True)

    return trainingDataset



def ApplyTrasformation(trainingDataset,typeOfDs):

        #listRank contains Ranks from ace to king
        #listSuits contains how many suits there are for any group
        #G label
        Allrows = []
        for index,rows in trainingDataset.iterrows():
            
            listRank=[0,0,0,0,0,0,0,0,0,0,0,0,0]
            listSuit=[0,0,0,0]
            listLabel=[0]
            listWinning=[0]


            for item in rows.items():
                KindOfCard = str(item[0])
                value = item[1]


                if(KindOfCard.startswith('S')):
                    value=value-1
                    listSuit[value] =  listSuit[value] + 1

                elif(KindOfCard.startswith('R')):
                    value=value-1
                    listRank[value] = 1
                elif(KindOfCard.startswith('G')):
                    listLabel[0] = value
                else:
                    listWinning[0] = value
                
            tmpAggregator = listRank+listSuit+listLabel+listWinning
            #print(tmpAggregator)
            Allrows.append(tmpAggregator)
        #print(len(Allrows))

        newColumns = ['Asso', 'Due', 'Tre', 'Quattro', 'Cinque', 'Sei', 'Sette', 'Otto',
                      'Nove', 'Dieci', 'Principe','Regina','Re','rankCuori','rankPicche','rankQuadri','rankFiori','label','isWinning']

        if(not "sample" in typeOfDs):
            encodedDf = pd.DataFrame(Allrows, columns=newColumns).drop_duplicates()
        else:
            encodedDf = pd.DataFrame(Allrows, columns=newColumns)
        
        print(encodedDf)
        print(encodedDf.shape)
        
        with open("./bin/resources/"+typeOfDs+"_encodedDf.pickle", 'wb') as output:
            pickle.dump(encodedDf, output)
        return encodedDf


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

    classesMetrics=['0','1']

    print("TREE Metric")
    print(classification_report(y_test, predictions, target_names=classesMetrics))
    PlotTrainErrors(X_train,y_train,clf,"TREE",activeEncoded)

    #Plotting tree
    plt.figure(figsize=(12,12))
    plot_tree(clf, fontsize=6)
    plt.savefig('./Images/treePlot.png', dpi=100)


def RandomForest(X_train,y_train,X_test,y_test,activeEncoded):
    clf = RandomForestClassifier(max_depth=None, random_state=0,n_estimators=1000,min_samples_leaf=3,min_samples_split=2,max_features='sqrt')
    clf = clf.fit(X_train,y_train)

    predictions = clf.predict(X_test)
    print(TREE+":ACC",accuracy_score(y_test, predictions))
    classes=np.unique(y_test)

    plt.close()
    cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
    plot_confusion_matrix(cm,classes,"TREE",activeEncoded)


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

    classesMetrics=['0','1']

    print("BAYES report classification")
    print(classification_report(y_test, predictions, target_names=classesMetrics,zero_division=1))

    print("Plotting train error bayes....")
    PlotTrainErrors(X_train,y_train,clf,BAYES,activeEncoded)

############################################################################


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

    classesMetrics=['0','1']

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

