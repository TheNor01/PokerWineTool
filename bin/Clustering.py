from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import silhouette_score
from utility.UtilityFunctions import ReadDataset
from sklearn.cluster import DBSCAN
#from yellowbrick.cluster import SilhouetteVisualizer

from utility.UtilityFunctions import plot_confusion_matrix,PlotTrainErrors

import warnings
warnings.filterwarnings("ignore")


def PrintShapeGraph(dataset):
    print("SIZE OF : (Records,Features)")
    print(dataset.shape)

    g_classes = len(set(dataset['label'].values))  # count distinct values label
    poker_hands = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    #poker_hands = [1, 2, 3, 4, 5, 6, 7, 8, 9]


    print(g_classes)
    cls = {}
    for i in range(g_classes):
        cls[i] = len(dataset[dataset.label==i])
    print(cls)

    #classes are unbalanced

    #Plot histogram

    plt.bar(poker_hands, [cls[i] for i in poker_hands], align='center')
    plt.xlabel('classes id')
    plt.ylabel('Number of instances')
    plt.title("Classes of Dataset")
    plt.show()

def Kmeans(X_train,X_test,y_test,activeEncoded):
    #We presume there are 11 K points (groups), equals to label


    #print("Computing KMENAS ENC: "+str(activeEncoded))
    clusters=range(1,11)
    #meandist=[]
    clusterSum=[]

    
    for k in clusters:
        model=KMeans(n_clusters=k,init='k-means++',)
        model.fit(X_train)
        clusterSum.append(model.inertia_)
        #clusassign=model.predict(X_train)


        #meandist.append(sum(np.min(cdist(X_train, model.cluster_centers_, 'braycurtis'), axis=1)) / X_train.shape[0])

    plt.plot(clusters, clusterSum)
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum square cluster')
    plt.title('Selecting k with the Elbow Method') # pick the fewest number of cluster
    plt.show()

    #Changing distance? https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html 

    #Seems there is no a clear elbow, braycurtis or cosine
    clusterNumbers = 10
    classes=np.unique(y_test)
    """
    clf=KMeans(n_clusters=clusterNumbers)
    clf.fit(X_train) # has cluster assingments based on using 3 clusters

    predictions = clf.predict(X_test)

    plt.close()

    cm = confusion_matrix(y_test, predictions, labels=classes)
    plot_confusion_matrix(cm,classes,"KMEANS",activeEncoded)
    """

    #K medoids in order to improve kmeans #page 399, cause a value range "is known"
    kmedoids = KMedoids(n_clusters=clusterNumbers).fit(X_train)
    predictions=kmedoids.predict(X_test)

    classesMetrics=['1','2','3','4','5','6','7','8','9']

    cm = confusion_matrix(y_test, predictions, labels=classes)
    plot_confusion_matrix(cm,classes,"KMETOIDS",activeEncoded)
    print(classification_report(y_test, predictions, target_names=classesMetrics))


def ClusteringGerc(X_train,X_test,y_test,activeEncoded):

    clusterFromDendrogam = 11
    classes=np.unique(y_test)

    print(classes)

    clustering_agglomerate = AgglomerativeClustering(clusterFromDendrogam).fit(X_train)
    predictions=clustering_agglomerate.fit_predict(X_test)

    classesMetrics=[1,2,3,4,5,6,7,8,9]

    cm = confusion_matrix(y_test, predictions, labels=classes)
    plot_confusion_matrix(cm,classes,"Agglomerative",activeEncoded)
    #print(classification_report(y_test, predictions, target_names=classesMetrics))


def ApplyPCA(X_train):
    pca = PCA(n_components = 0.70) 
    trasformedDf = pca.fit_transform(X_train)

    print("cumsum variance pca")
    print(pca.explained_variance_ratio_.cumsum())


    # plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model.labels_,) # plot 1st canonical variable on x axis, 2nd on y-axis
    # plt.xlabel('Canonical variable 1')
    # plt.ylabel('Canonical variable 2')
    # plt.title('Scatterplot of Canonical Variables for '+str(clusterNumbers)+' Clusters')
    # plt.show()

    PC_values = np.arange(pca.n_components_)+1
    plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.show()

    return pca,trasformedDf
    

if __name__ == "__main__":

    print("CLUSTERING")

    #trainingDataset = ReadDataset("./bin/resources/poker-hand-training-true.data")
    #testingDataset = ReadDataset("./bin/resources/poker-hand-testing.data")

    

    with open('./bin/resources/training-sampled_lower_encodedDf.pickle', 'rb') as data:
        trainingDataset_encoded = pickle.load(data).astype(int)

    with open('./bin/resources/testing_encodedDf.pickle', 'rb') as data:
        testingDataset_encoded = pickle.load(data).astype(int)




    PrintShapeGraph(trainingDataset_encoded)


    
    #How can we see them in a 2 space? Maybe a linear combination?
     

    #Semi supervised approch, we know how many labels there are

     
    #K means
    #Cluster gerarchico?


    #remove 0 label rows
    X_train_encoded =  trainingDataset_encoded[(trainingDataset_encoded.label != 0)].copy()

    print(X_train_encoded.shape)

    y_train_df  =  X_train_encoded

    X_train_encoded = X_train_encoded.loc[:, X_train_encoded.columns != 'label']
    X_train_encoded = X_train_encoded.drop('isWinning', axis=1)
    y_train_encoded = y_train_df.loc[:, y_train_df.columns == 'label'].values.ravel()



    #remove 0 label rows
    X_test_encoded =  trainingDataset_encoded[(trainingDataset_encoded.label != 0)].copy()

    y_test_df = X_test_encoded

    X_test_encoded = X_test_encoded.loc[:, X_test_encoded.columns != 'label']
    y_test_encoded = y_test_df.loc[:, y_test_df.columns == 'label'].values.ravel()



    

    """

    #pca unicode
    print("PCA ENCODED")
    pca,X_train_encoded_pca = ApplyPCA(X_train_encoded)
    X_test_encoded_pca = pca.transform(X_test_encoded)

    print(X_train_encoded_pca.shape)
    print(X_test_encoded_pca.shape)

    
    components = X_test_encoded_pca.shape[1]
    #print(components)
    newColumns = ['P'+str(item) for item in range(1, components+1)]
    X_test_encoded_pca.columns = newColumns
    """

    #do the same with encoded
    #Kmeans(X_train_encoded_pca,X_test_encoded_pca,y_test_encoded,1)


    #plt.figure(figsize =(8, 8))
    #plt.title('Visualising the data')
    clusterFromDendrogam = 11
    #plt.show()

    
    plt.cla()
    plt.close()


    
    ClusteringGerc(X_train_encoded,X_test_encoded,y_test_encoded,1)

    exit()



    #clustering gerarchico comparison, anche indice di Dunn
    


    plt.close()
    #EVALUATE CLUSTERING
    k = [2,3, 4, 5, 6,7,8,9,10]
    # Appending the silhouette scores of the different models to the list
    silhouette_scores = []

    for cluster in k:
        clustering_agglomerate = AgglomerativeClustering(cluster).fit(X_train_encoded_pca)
        silhouette_scores.append(silhouette_score(X_train_encoded_pca, clustering_agglomerate.fit_predict(X_train_encoded_pca)))


    plt.bar(k, silhouette_scores)
    plt.xlabel('Number of clusters', fontsize = 20)
    plt.ylabel('S(i)', fontsize = 20)
    plt.show()