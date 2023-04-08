from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import itertools
from sklearn.model_selection import ShuffleSplit,LearningCurveDisplay


def plot_confusion_matrix(cm, classes, classifier,normalize=False, cmap=cm.Blues, png_output="./Images", show=True):
       
        title='Confusion matrix OF '+classifier
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
            f.savefig(os.path.join(png_output,'confusion_matrix_'+classifier+'.png'), bbox_inches='tight')

        if show:
            plt.show()
            plt.close(f)
        else:
            plt.close(f)

def PlotTrainErrors(X_train,y_train,classifier):
    plt.close()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), sharey=True)
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

    for ax_idx, estimator in enumerate([classifier]):
        LearningCurveDisplay.from_estimator(estimator, **common_params, ax=ax[ax_idx])
        handles, label = ax[ax_idx].get_legend_handles_labels()
        ax[ax_idx].legend(handles[:2], ["Training Score", "Test Score"])
        ax[ax_idx].set_title(f"Learning Curve for {estimator.__class__.__name__}")

        plt.show()


def ReadDataset(path):
    features = np.array(['S1', 'R1','S2', 'R2','S3', 'R3','S4', 'R4','S5','R5','G'])
    columnsFeatures = pd.Series(features)
    trainingDataset = pd.read_csv(path,names=columnsFeatures)
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


            for item in rows.items():
                KindOfCard = str(item[0])
                value = item[1]
                if(KindOfCard.startswith('S')):
                    value=value-1
                    listSuit[value] =  listSuit[value] + 1

                elif(KindOfCard.startswith('R')):
                    value=value-1
                    listRank[value] = 1
                else:
                    listLabel[0] = value


            tmpAggregator = listRank+listSuit+listLabel
            print(tmpAggregator)
            Allrows.append(tmpAggregator)
        print(len(Allrows))

        newColumns = ['Asso', 'Due', 'Tre', 'Quattro', 'Cinque', 'Sei', 'Sette', 'Otto'
                        ,'' 'Nove', 'Dieci', 'Principe','Regina','Re','rankCuori','rankPicche','rankQuadri','rankFiori','label']

        encodedDf = pd.DataFrame(Allrows, columns=newColumns)
        print(encodedDf)
        print(encodedDf.shape)
        
        with open("./bin/resources/"+typeOfDs+"_encodedDf.pickle", 'wb') as output:
            pickle.dump(encodedDf, output)
        return encodedDf