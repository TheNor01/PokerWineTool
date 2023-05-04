#Todo

#12 attributes

"""
   For more information, read [Cortez et al., 2009].

   Input variables (based on physicochemical tests):
   1 - fixed acidity
   2 - volatile acidity
   3 - citric acid
   4 - residual sugar
   5 - chlorides
   6 - free sulfur dioxide
   7 - total sulfur dioxide
   8 - density
   9 - pH
   10 - sulphates
   11 - alcohol
   Output variable (based on sensory data): 
   12 - quality (score between 0 and 10)
"""

#maybe we can give as output red or white instead of quality

"""
red 0
white 1

"""

import pickle
import csv
import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn
import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression,RidgeCV
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

rows = []

def CheckIntegrityDataset(dataset):
    #Method used to detect strange values, according to problem specification

    print("Checking null or NAN values...")

    checkNan = dataset.isnull().values.any() #returns true if there is one true at least
    print(checkNan)

#     print("Checking range goal (0,9) column")
#     checkRange2 = dataset["quality"].between(0,10).values.any()
#     print(checkRange2)

    print("Checking duplicates...")
    dataset.drop_duplicates(subset=None,inplace=True)

def PrintShapeGraph(dataset):

    print("SET VALUES")
    print(set(dataset['wineLabel'].values))

    quality_classes = 2  # count distinct values label
    wine_label = [0, 1]

    cls = {}
    for i in range(quality_classes):
        cls[i] = len(dataset[dataset.wineLabel==i])
    print(cls)


    #Plot histogram, change scale
    plt.bar(wine_label, [cls[i] for i in wine_label], align='center')
    plt.xlabel('classes id')
    plt.ylabel('Number of instances')
    plt.title("Classes of Dataset")
    plt.show()



print(os.getcwd())


#read red wines
dataWineRed= pd.read_csv("./bin/resources/wine/winequality-red.csv",sep=';')

print(dataWineRed)
print(dataWineRed.columns)
print(dataWineRed.shape)

dataWineRed["wineLabel"] = 0

dataWineWhite= pd.read_csv("./bin/resources/wine/winequality-white.csv",sep=';')

print(dataWineWhite)
print(dataWineWhite.columns)
print(dataWineWhite.shape)

dataWineWhite["wineLabel"] = 1

print("merge")
unionWine = dataWineRed.merge(dataWineWhite, how="outer").copy()
print(unionWine.shape)

unionWine.drop('quality', axis=1, inplace=True)


print(unionWine)

CheckIntegrityDataset(unionWine)
PrintShapeGraph(unionWine)

#we can see a different range min-max for every column, in particular alcohol
print(unionWine.describe())


#let's zscore, we can use it cause 75%percentile is near to mean value. also STD
print("zSCORE")
X_train_scored = pd.DataFrame()
for col in unionWine.columns:
        if(col=="wineLabel"): 
                X_train_scored[str(col)] = unionWine[col]
        else:
                X_train_scored[str(col)] = zscore(unionWine[col])


print(X_train_scored)
print(X_train_scored.describe())

#check covariance matrice
#https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.cov.html
cov_matrix = pd.DataFrame.cov(unionWine)
sn.heatmap(cov_matrix, annot=False, fmt='g')
plt.show()


#Correlation pearson
fig, ax = plt.subplots(figsize=(10, 6))


pearsonDf = unionWine.corr().abs()
sn.heatmap(pearsonDf, ax=ax, annot=True)
plt.title("pearson CORR")
plt.show()


kendallDf = unionWine.corr(method='kendall').abs()
fig, ax = plt.subplots(figsize=(10, 6))
sn.heatmap(kendallDf, ax=ax, annot=True)
plt.title("kendall CORR")
plt.show()

s = pearsonDf.unstack()
so = s.sort_values(kind="quicksort")

s1 = kendallDf.unstack()
so1 = s1.sort_values(kind="quicksort")

print(so[(so >= 0.50) & (so < 1.00)])
print("-----------")
print(so1[(so1 >= 0.50) & (so1 < 1.00)])

colors = ['#E69F00', '#56B4E9']
names= ['red','white']
#plot different chart for two label
#for col in unionWine.columns:


"""
for s in unionWine.columns:
# plotting both distibutions on the same figure
        sn.scatterplot(data=unionWine, x='pH',y=s, hue='wineLabel',alpha=0.5)
        plt.show()
"""

y_train_scored = X_train_scored.loc[:, X_train_scored.columns == 'wineLabel'].values.ravel()

X_train, X_test, y_train, y_test = train_test_split(X_train_scored, y_train_scored, test_size = 0.2, random_state = 42)


X_test.drop('wineLabel', axis=1, inplace=True)
X_train.drop('wineLabel', axis=1, inplace=True)


print(X_test)
print(y_test)
"""
REGRESSION LINEAR


"""

regressorLinear = LinearRegression()
regressorLinear.fit(X_train,y_train)

y_predective_linear = regressorLinear.predict(X_test)
score = r2_score(y_test,y_predective_linear)*100
print("R2 Score original",score)

print('MAE:', mean_absolute_error(y_test, y_predective_linear))
print('MSE:', mean_squared_error(y_test, y_predective_linear))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_predective_linear)))