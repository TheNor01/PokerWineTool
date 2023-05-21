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

    classesMetrics=['0','1','2','3','4','5','6','7','8','9']

    cm = confusion_matrix(y_test, predictions, labels=classes)
    plot_confusion_matrix(cm,classes,"KMETOIDS",activeEncoded)
    print(classification_report(y_test, predictions, target_names=classesMetrics))


def ClusteringGerc(X_train,X_test,y_test,activeEncoded):

    clusterFromDendrogam = 11
    classes=np.unique(y_test)
    clustering_agglomerate = DBSCAN(clusterFromDendrogam).fit(X_train)
    predictions=clustering_agglomerate.fit_predict(X_test)

    classesMetrics=['0','1','2','3','4','5','6','7','8','9']

    cm = confusion_matrix(y_test, predictions, labels=classes)
    plot_confusion_matrix(cm,classes,"Agglomerative",activeEncoded)
    print(classification_report(y_test, predictions, target_names=classesMetrics))


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

    trainingDataset = ReadDataset("./bin/resources/poker-hand-training-true.data")
    testingDataset = ReadDataset("./bin/resources/poker-hand-testing.data")



    with open('./bin/resources/training_encodedDf.pickle', 'rb') as data:
        trainingDataset_encoded = pickle.load(data)

    with open('./bin/resources/testing_encodedDf.pickle', 'rb') as data:
        testingDataset_encoded = pickle.load(data)


    
    #How can we see them in a 2 space? Maybe a linear combination?
     

    #Semi supervised approch, we know how many labels there are

     
    #K means
    #Cluster gerarchico?


    
   
    X_train_encoded = trainingDataset_encoded.loc[:, trainingDataset_encoded.columns != 'label']
    X_train_encoded = X_train_encoded.drop('isWinning', axis=1)
    y_train_encoded = trainingDataset_encoded.loc[:, trainingDataset_encoded.columns == 'label'].values.ravel()


    print(testingDataset_encoded[testingDataset_encoded.label == 9])



    X_test_encoded = testingDataset_encoded.loc[:, testingDataset_encoded.columns != 'label']
    y_test_encoded = testingDataset_encoded.loc[:, testingDataset_encoded.columns == 'label'].values.ravel()



    

    #pca unicode
    print("PCA ENCODED")
    pca,X_train_encoded_pca = ApplyPCA(X_train_encoded)
    X_test_encoded_pca = pca.transform(X_test_encoded)

    print(X_train_encoded_pca.shape)
    print(X_test_encoded_pca.shape)

    
    """
    components = X_test_encoded_pca.shape[1]
    #print(components)
    newColumns = ['P'+str(item) for item in range(1, components+1)]
    X_test_encoded_pca.columns = newColumns
    """

    #do the same with encoded
    #Kmeans(X_train_encoded_pca,X_test_encoded_pca,y_test_encoded,1)


    plt.figure(figsize =(8, 8))
    plt.title('Visualising the data')
    clusterFromDendrogam = 11
    #plt.show()

    
    plt.cla()
    plt.close()

    
    ClusteringGerc(trainingDataset,testingDataset,y_test_encoded,1)



    #clustering gerarchico comparison, anche indice di Dunn
    

    exit()

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