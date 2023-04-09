from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans,AgglomerativeClustering
import scipy.cluster.hierarchy as shc
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances,silhouette_score
from utility.UtilityFunctions import ReadDataset

def Kmeans(X_train,X_test):
    #We presume there are 11 K points, equals to label
    clusters=range(1,11)
    meandist=[]

    
    for k in clusters:
        model=KMeans(n_clusters=k)
        model.fit(X_train)
        clusassign=model.predict(X_train)
        meandist.append(sum(np.min(cdist(X_train, model.cluster_centers_, 'braycurtis'), axis=1)) / X_train.shape[0])

    plt.plot(clusters, meandist)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average distance')
    plt.title('Selecting k with the Elbow Method') # pick the fewest number of cluster
    plt.show()

    #Changing distance? https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html 

    #Seems there is no a clear elbow, braycurtis or cosine
    clusterNumbers = 3
    model=KMeans(n_clusters=clusterNumbers)
    model.fit(X_train) # has cluster assingments based on using 3 clusters

    score = accuracy_score(y_test,model.predict(X_test))
    print('Accuracy:{0:f}'.format(score))

    #K medoids in order to improve kmeans #page 399
    kmedoids = KMedoids(n_clusters=clusterNumbers).fit(X_train)
    score = accuracy_score(y_test,kmedoids.predict(X_test))
    print('Accuracy Medoids:{0:f}'.format(score))

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
    #plt.show()

    print(pca.explained_variance_ratio_)

    return X_train_pca
    

def Kmetoids():
    pass

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
     
    #K nn
    #K means
    #Graph based?

    X_train = trainingDataset[['S1', 'R1','S2', 'R2','S3', 'R3','S4', 'R4','S5','R5']]
    y_test = testingDataset['G']

    y_test = testingDataset['G']
    X_test = testingDataset[['S1', 'R1','S2', 'R2','S3', 'R3','S4', 'R4','S5','R5']]

    Kmeans(X_train,X_test)

    #Variance 
    print("VARIANCE DATASET")
    print(X_train.var())

    #Maybe we can reduce features?
    X_train_pca = ApplyPCA(X_train)

    X_train_pca=pd.DataFrame(X_train_pca,).copy()

    components = X_train_pca.shape[1]

    print(components)
    newColumns = ['P'+str(item) for item in range(1, components+1)]

    X_train_pca.columns = newColumns

    print(X_train_pca)

    plt.figure(figsize =(8, 8))
    plt.title('Visualising the data')
    Dendrogram = shc.dendrogram((shc.linkage(X_train_pca, method ='ward')))

    plt.show()


    plt.cla()
    plt.close()

    #Choose k CLUSTER - DRAW  line in order to seperate 

    clustering_agglomerate = AgglomerativeClustering(5).fit(X_train_pca)
    plt.figure(figsize =(6, 6))
    plt.scatter(X_train_pca['P1'], X_train_pca  ['P2'],
                c = clustering_agglomerate.fit_predict(X_train_pca) , cmap ='rainbow')
    plt.show()

    #clustering gerarchico comparison, anche indice di Dunn



    k = [2, 3, 4, 5, 6, 8]
    
    # Appending the silhouette scores of the different models to the list
    silhouette_scores = []

    for cluster in k:
        clustering_agglomerate = AgglomerativeClustering(cluster).fit(X_train_pca)
        silhouette_scores.append(silhouette_score(X_train_pca, clustering_agglomerate.fit_predict(X_train_pca)))


    plt.bar(k, silhouette_scores)
    plt.xlabel('Number of clusters', fontsize = 20)
    plt.ylabel('S(i)', fontsize = 20)
    plt.show()