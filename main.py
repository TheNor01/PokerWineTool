#Todo

#solve common cards 
#recall evalutatiom

#Evaluate cluster -> dunn index


#Balance dataset --> page 413
    #can we delete 0 label ?
    #data weighting for svm -> set as balanced
    #clustering??

import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

with open('./bin/resources/training_encodedDf.pickle', 'rb') as data:
        trainingDataset_encoded = pickle.load(data)


print(trainingDataset_encoded)

print(trainingDataset_encoded.describe())

cov_matrix = pd.DataFrame.cov(trainingDataset_encoded)
sn.heatmap(cov_matrix, annot=False, fmt='g')
plt.show()


quit()

for col in trainingDataset_encoded.columns:
        trainingDataset_encoded[col].plot(kind='kde')
        plt.title(str(col))
        plt.show()


