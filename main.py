#Todo

#solve common cards 
#recall evalutatiom

#Evaluate cluster -> dunn index


#Balance dataset --> page 413
    #can we delete 0 label ?
    #clustering??
#plot 18D?

import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from sklearn.tree import DecisionTreeClassifier





with open('./bin/resources/training-sampled-dropped_encodedDf.pickle', 'rb') as data:
        droppedTR_encoded = pickle.load(data)



print(droppedTR_encoded)

X_train_sampled_encoded = droppedTR_encoded.loc[:, droppedTR_encoded.columns != 'label']
y_train_sampled_encoded = droppedTR_encoded.loc[:, droppedTR_encoded.columns == 'label'].values.ravel()


clf = DecisionTreeClassifier(criterion='gini',max_depth=6,class_weight='balanced')
clf = clf.fit(X_train_sampled_encoded,y_train_sampled_encoded)

predictions = clf.predict(X_test)
print(TREE+":ACC",accuracy_score(y_test, predictions))
classes=np.unique(y_test)

plt.close()
cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
plot_confusion_matrix(cm,classes,"TREE",activeEncoded)



exit()


with open('./bin/resources/training_encodedDf.pickle', 'rb') as data:
        trainingDataset_encoded = pickle.load(data)


print(trainingDataset_encoded)

print(trainingDataset_encoded.describe())

cov_matrix = pd.DataFrame.cov(trainingDataset_encoded)
sn.heatmap(cov_matrix, annot=False, fmt='g')
plt.show()


with open('./bin/resources/training-sampled_encodedDf.pickle', 'rb') as data:
        trainingDataset_sampled_encoded = pickle.load(data)
        filterDf =  trainingDataset_sampled_encoded[trainingDataset_sampled_encoded.label != 0]

        print(filterDf)



X_train_sampled_encoded = filterDf.loc[:, filterDf.columns != 'label']
y_train_sampled_encoded = filterDf.loc[:, filterDf.columns == 'label'].values.ravel()



pca = PCA(n_components = 3) 
trasformedDf = pca.fit_transform(X_train_sampled_encoded)

Xax = trasformedDf[:,0]
Yax = trasformedDf[:,1]
Zax = trasformedDf[:,2]

print(trasformedDf)


cdict = {0:'#1f77b4',1:'#ff7f0e',2:'#2ca02c',3:'#d62728',4:'#9467bd',5:'#8c564b',6:'#e377c2',7:'#7f7f7f',8:'#bcbd22',9:'#17becf'}
label = {0:'nothing',1:'coppia',2:'doppiaCoppia',3:'tris',4:'scala',5:'colore',6:'full',7:'poker',8:'scalaColore',9:'scalaReale'}

fig = plt.figure(figsize=(14,9))
ax = fig.add_subplot(111, projection='3d')

for l in np.unique(y_train_sampled_encoded):
 ix=np.where(y_train_sampled_encoded==l)
 ax.scatter(Xax[ix], Yax[ix], Zax[ix], c=cdict[l], s=60,label=label[l])


ax.set_xlabel("PC1", fontsize=12)
ax.set_ylabel("PC2", fontsize=12)
ax.set_zlabel("PC3", fontsize=12)
 
ax.view_init(30, 125)
ax.legend()
plt.title("3D PCA plot")
plt.show()

quit()

for col in trainingDataset_encoded.columns:
        trainingDataset_encoded[col].plot(kind='kde')
        plt.title(str(col))
        plt.show()


