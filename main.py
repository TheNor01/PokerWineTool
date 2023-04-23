#Todo

#solve common cards 
#recall evalutatiom

#Evaluate cluster -> dunn index


#Balance dataset --> page 413
    #can we delete 0 label ?
    #data weighting for svm
    #clustering??

import pickle


with open('./bin/resources/training_encodedDf.pickle', 'rb') as data:
        trainingDataset_encoded = pickle.load(data)


print(trainingDataset_encoded)
print(type(trainingDataset_encoded))
print(trainingDataset_encoded.shape)

df2=trainingDataset_encoded.drop_duplicates()
print(df2.shape)