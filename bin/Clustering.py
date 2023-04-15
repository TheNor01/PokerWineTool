from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans,AgglomerativeClustering
import scipy.cluster.hierarchy as shc
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import pairwise_distances,silhouette_score
from utility.UtilityFunctions import ReadDataset
from yellowbrick.cluster import SilhouetteVisualizer

from utility.UtilityFunctions import plot_confusion_matrix,PlotTrainErrors

import warnings
warnings.filterwarnings("ignore")

def Kmeans(X_train,X_test,y_test,activeEncoded):
    #We presume there are 11 K points (groups), equals to label


    print("Computing KMENAS ENC: "+str(activeEncoded))


    clusters=range(1,11)
    meandist=[]
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
    clusterNumbers = 3
    clf=KMeans(n_clusters=clusterNumbers)
    clf.fit(X_train) # has cluster assingments based on using 3 clusters

    predictions = clf.predict(X_test)
    classes=np.unique(y_test)

    plt.close()
    cm = confusion_matrix(y_test, predictions, labels=classes)
    plot_confusion_matrix(cm,classes,"KMEANS",activeEncoded)

    #K medoids in order to improve kmeans #page 399, cause a value range "is known"
    kmedoids = KMedoids(n_clusters=clusterNumbers).fit(X_train)
    predictions=kmedoids.predict(X_test)

    classesMetrics=['0','1','2','3','4','5','6','7','8','9']

    print(classification_report(y_test, predictions, target_names=classesMetrics))

def ApplyPCA(X_train):
    pca = PCA(n_components = 0.90) 
    X_train_pca = pca.fit_transform(X_train)

    print("Shape pca")
    print(X_train_pca.shape)

    #https://plotly.com/python/pca-visualization/

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

    return X_train_pca
    

if __name__ == "__main__":

    print("CLUSTERING")

    trainingDataset = ReadDataset("./bin/resources/poker-hand-training-true.data")
    testingDataset = ReadDataset("./bin/resources/poker-hand-testing.data")

    with open('./bin/resources/training_encodedDf.pickle', 'rb') as data:
        trainingDataset_encoded = pickle.load(data)

    with open('./bin/resources/testing_encodedDf.pickle', 'rb') as data:
        testingDataset_encoded = pickle.load(data)
    #https://jasminedaly.com/2016-05-25-kmeans-analysis-in-python/

    #How can we see them in a 2 space? Maybe a linear combination?
     

    #Semi supervised approch, we know how many labels there are

     
    #K means
    #Cluster gerarchico?

    X_train = trainingDataset[['S1', 'R1','S2', 'R2','S3', 'R3','S4', 'R4','S5','R5']]
    y_train = trainingDataset['G']

    y_test = testingDataset['G']
    X_test = testingDataset[['S1', 'R1','S2', 'R2','S3', 'R3','S4', 'R4','S5','R5']]

    X_train_encoded = trainingDataset_encoded.loc[:, trainingDataset_encoded.columns != 'label']
    y_train_encoded = trainingDataset_encoded.loc[:, trainingDataset_encoded.columns == 'label'].values.ravel()


    X_test_encoded = testingDataset_encoded.loc[:, testingDataset_encoded.columns != 'label']
    y_test_encoded = testingDataset_encoded.loc[:, testingDataset_encoded.columns == 'label'].values.ravel()

    #Apply kmeans
    #Kmeans(X_train,X_test,y_test,0)

    #Variance 
    print("PCA")
    #print(X_train.var())

    #can we reduce features?
    X_train_pca = ApplyPCA(X_train)
    X_test_pca = ApplyPCA(X_test)

    #pca unicode


    print("PCA ENCODED")
    X_train_encoded_pca = ApplyPCA(X_train_encoded)
    X_test_encoded_pca = ApplyPCA(X_test_encoded)


    X_train_pca=pd.DataFrame(X_train_pca,).copy()
    X_test_pca=pd.DataFrame(X_test_pca,).copy()
    components = X_train_pca.shape[1]

    #print(components)
    newColumns = ['P'+str(item) for item in range(1, components+1)]

    X_train_pca.columns = newColumns

    #print(X_train_pca)
    Kmeans(X_train_pca,X_test_pca,y_test,0)


    #do the same with encoded
    Kmeans(X_train_encoded,X_test_encoded,y_test_encoded,1)

  
    #Choose k CLUSTER - DRAW  line in order to seperate 

    plt.figure(figsize =(8, 8))
    plt.title('Visualising the data')

    #method could be changed
    #Dendrogram = shc.dendrogram((shc.linkage(X_train_pca, method ='ward')))

    clusterFromDendrogam = 11
    #plt.show()

    """
    plt.cla()
    plt.close()
    clustering_agglomerate = AgglomerativeClustering(clusterFromDendrogam).fit(X_train_encoded)
    
    plt.figure(figsize =(6, 6))
    plt.scatter(X_train_encoded['P1'], X_train_encoded  ['P2'],
                c = clustering_agglomerate.fit_predict(X_train_encoded) , cmap ='rainbow')
    plt.show()

    #clustering gerarchico comparison, anche indice di Dunn
    """

    
    plt.close()
    #EVALUATE CLUSTERING
    k = [2, 3, 4, 5, 6, 8]
    # Appending the silhouette scores of the different models to the list
    silhouette_scores = []

    for cluster in k:
        clustering_agglomerate = AgglomerativeClustering(cluster).fit(X_train_encoded)
        silhouette_scores.append(silhouette_score(X_train_encoded, clustering_agglomerate.fit_predict(X_train_encoded)))


    plt.bar(k, silhouette_scores)
    plt.xlabel('Number of clusters', fontsize = 20)
    plt.ylabel('S(i)', fontsize = 20)
    plt.show()