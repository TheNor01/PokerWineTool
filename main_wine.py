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
from sklearn.feature_selection import SequentialFeatureSelector,SelectKBest,chi2
from scipy.stats import zscore
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression,RidgeCV
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

rows = []

def CheckIntegrityDataset(dataset):
    #Method used to detect strange values, according to problem specification

    print("Checking null or NAN values...")

    checkNan = dataset.isnull().values.any() #returns true if there is one true at least
    print(checkNan)

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
    plt.xticks([0,1])
    plt.xlabel('classes id')
    plt.ylabel('Number of instances')
    plt.title("Classes of Dataset")
    plt.show()


def calculateStatsModel(y_test,y_predective_linear):
    score = r2_score(y_test,y_predective_linear)
    print("R2 Score original",score)

    print("R2 ADJ with p regressor")
    print(1-(1-score)*((len(X_test)-1)/(len(X_test)-len(X_test.columns)-1)))

    print('MAE LR:', mean_absolute_error(y_test, y_predective_linear))
    print('MSE LR:', mean_squared_error(y_test, y_predective_linear))
    print('RMSE LR:', np.sqrt(mean_squared_error(y_test, y_predective_linear)))


def plotTrainTestError(fit_results,test_pred_results):
    #### SCATTER PLOT - TRAIN
    plt.scatter(fit_results.index,  fit_results['Y_TRUE'], alpha = 0.4, s = 12, label = 'Y_TRUE')
    plt.scatter(fit_results.index,  fit_results['Y_FIT'], alpha = 0.4, s = 12, label = 'Y_FIT')
    plt.xlabel('Y_TRUE_RANK (SORTED)')
    plt.ylabel('wine label')
    plt.title('TRAIN SET')
    plt.legend()

    plt.show()

    plt.cla()
    #### SCATTER PLOT - TEST
    plt.scatter(test_pred_results.index,  test_pred_results['Y_TRUE'], alpha = 0.6, s = 12, label = 'Y_TRUE')
    plt.scatter(test_pred_results.index,  test_pred_results['Y_PRED'], alpha = 0.4, s = 12, label = 'Y_PRED')
    plt.xlabel('Y_TRUE_RANK (SORTED)')
    plt.ylabel('wine label')
    plt.title('TEST SET')
    plt.legend()
    plt.show()


def plotResiduals(y_test,y_predective_linear):
    residuals = y_test - y_predective_linear
    print("Residuals",residuals)
    print("predective ",y_predective_linear)

    plt.scatter(y_predective_linear, residuals)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('Residual Plot')
    plt.show()

    plt.cla()
    plt.close()

    #ok, it follows a gaussian, not a clear one
    plt.hist(residuals, bins=10)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals')
    plt.show()


    print(type(residuals))
    print("AVG residuals",residuals.mean())
    print("VARIANCE residuals",residuals.var())

def DoBackWardAutomatic(X_train,y_train):
    feature_names = np.array(X_train.columns)
    backWard_model_linear = SequentialFeatureSelector(regressorLinear, n_features_to_select=5,direction='backward').fit(X_train, y_train)
    print("Features selected by backward sequential selection: "f"{feature_names[backWard_model_linear.get_support()]}")
    
    return feature_names[backWard_model_linear.get_support()]
    #X_new = SelectKBest(chi2, k=5).fit_transform(X, y)


def makeDataframeResult(y_train,y_test,model,y_predective_linear,X_train):
    fit_results = pd.DataFrame(y_train)
    fit_results.columns = ['Y_TRUE']
    test_pred_results = pd.DataFrame(y_test)
    test_pred_results.columns = ['Y_TRUE']

    fit_results['Y_FIT'] = model.predict(X_train).ravel() ##test train error
    test_pred_results['Y_PRED'] = y_predective_linear # test test error

    fit_results = fit_results.sort_values(by = ['Y_TRUE']).reset_index(drop = True)
    test_pred_results = test_pred_results.sort_values(by = ['Y_TRUE']).reset_index(drop = True)

    return fit_results,test_pred_results

def ApplyPCA(data):
    pca = PCA(n_components = 0.90) 
    #pca = PCA(n_components = 2) 
    trasformedDf = pca.fit_transform(data)

    print("cumsum variance pca")
    #7 features explain 90% variance
    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
    print(exp_var_cumul)


    fig = px.area(
        x=range(1, exp_var_cumul.shape[0] + 1),
        y=exp_var_cumul,
        labels={"x": "# Components", "y": "Explained Variance"}
    )

    #fig.show()
    
    
    #plt.cla()

    PC_values = np.arange(pca.n_components_)+1
    plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    #plt.show()

    return pca,trasformedDf



def loadingPCA(pca,pca_dataframe,noLabel,originalDf):


    plt.cla()
    plt.close()
    features = list(noLabel.columns.values)

    print(features)

    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    print(loadings)
    fig = px.scatter(pca_dataframe, x=0, y=1, color=originalDf['wineLabel'])

    for i, feature in enumerate(features):
        fig.add_annotation(
            ax=0, ay=0,
            axref="x", ayref="y",
            x=loadings[i, 0],
            y=loadings[i, 1],
            showarrow=True,
            arrowsize=2,
            arrowhead=2,
            xanchor="right",
            yanchor="top"
        )
        fig.add_annotation(
            x=loadings[i, 0],
            y=loadings[i, 1],
            ax=0, ay=0,
            xanchor="center",
            yanchor="bottom",
            text=feature,
            yshift=5,
        )
    #fig.show()

print(os.getcwd())

#fare main

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


CheckIntegrityDataset(unionWine)
PrintShapeGraph(unionWine)

#we can see a different range min-max for every column, in particular alcohol
##print(dataWineRed.describe())
print(dataWineWhite.describe())
#print(unionWine.describe())



#unionWine.boxplot(column=['residual sugar'], return_type='axes')
#plt.show()


#let's zscore, we can use it cause 75%percentile is near to mean value. also STD
print("zSCORE")
X_train_scored = pd.DataFrame()
for col in unionWine.columns:
        if(col=="wineLabel"): 
                X_train_scored[str(col)] = unionWine[col]
        else:
                X_train_scored[str(col)] = zscore(unionWine[col])


#print(X_train_scored)
print(X_train_scored.describe())



X_train_scored.boxplot(column=['residual sugar'], return_type='axes')
#plt.show()

print(X_train_scored[X_train_scored['residual sugar']<=2.00])

X_train_scored = X_train_scored[X_train_scored['residual sugar']<=2.00].copy()



#check covariance matrice
#https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.cov.html
cov_matrix = pd.DataFrame.cov(X_train_scored)
#sn.heatmap(cov_matrix, annot=False,cbar=True, fmt='g')
#plt.title("COVARIANCE MATRIX")
#plt.show()


#Correlation pearson
fig, ax = plt.subplots(figsize=(10, 6))

pearsonDf = X_train_scored.corr().abs()
sn.heatmap(pearsonDf, ax=ax, annot=True)
#plt.title("pearson CORR")
#plt.show()



kendallDf = X_train_scored.corr(method='kendall').abs()
#fig, ax = plt.subplots(figsize=(10, 6))
sn.heatmap(kendallDf, ax=ax, annot=True)
#plt.title("kendall CORR")
#plt.show()


s = pearsonDf.unstack()
so = s.sort_values(kind="quicksort")

s1 = kendallDf.unstack()
so1 = s1.sort_values(kind="quicksort")


print("-----PEARSON------")
print(so[(so >= 0.50) & (so < 1.00)])
print("-----KENDALL------")
print(so1[(so1 >= 0.50) & (so1 < 1.00)])

plt.cla()
plt.clf()
plt.close()




y_train_scored = X_train_scored.loc[:, X_train_scored.columns == 'wineLabel'].values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X_train_scored, y_train_scored, test_size = 0.2, random_state = 42)


PrintShapeGraph(X_test)


X_test.drop('wineLabel', axis=1, inplace=True)
X_train.drop('wineLabel', axis=1, inplace=True)


#print(X_test)
#print(y_test)
"""
REGRESSION LINEAR
"""

#Multivariate LINEAR REGRESSION
#PAGE 168

regressorLinear = LinearRegression().fit(X_train,y_train)

y_predective_linear = regressorLinear.predict(X_test)

print("Y_TRAIN:",set(y_train))
print(set(y_predective_linear))

#Study errors


fit_results,test_pred_results = makeDataframeResult(y_train,y_test,regressorLinear,y_predective_linear,X_train)

plotTrainTestError(fit_results,test_pred_results)

score = r2_score(y_test,y_predective_linear)
print("R2 Score original",score)


plt.cla()


#Calculate residual average, variance

plotResiduals(y_test,y_predective_linear)


#Can we do a better score?

#try another linear regressior, or logistic

#OLS and F distribution

calculateStatsModel(y_test,y_predective_linear)




#FEATURE SELECTION with AIC consideration

print("BACKWARD SELECTION")

columns_selected = DoBackWardAutomatic(X_train,y_train).tolist()

print(type(columns_selected))

X_train_selected = X_train[columns_selected].copy()
X_test_selected = X_test[columns_selected].copy()

regressorLinear_sel = LinearRegression().fit(X_train_selected,y_train)

print("STATS SELECTED FEATURES MODEL ")
y_predective_linear = regressorLinear_sel.predict(X_test_selected)

#calculateStatsModel(y_test,y_predective_linear)
#plotResiduals(y_test,y_predective_linear)

fit_results,test_pred_results = makeDataframeResult(y_train,y_test,regressorLinear_sel,y_predective_linear,X_train_selected)

#plotTrainTestError(fit_results,test_pred_results)


print("\n\nLOGISTIC REGRESSION ")
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
logreg = LogisticRegression(dual=False,class_weight='balanced',multi_class='ovr').fit(X_train_selected,y_train)

print("STATS SELECTED FEATURES MODEL LOGISTIC ")
y_predective_log= logreg.predict(X_test_selected)

calculateStatsModel(y_test,y_predective_log)
#plotResiduals(y_test,y_predective_log)

fit_results,test_pred_results = makeDataframeResult(y_train,y_test,logreg,y_predective_log,X_train_selected)

#plotTrainTestError(fit_results,test_pred_results)

print("PCA ANALYSIS")

X_train_noLabel = X_train
X_test_noLabel = X_test

pcaModel,pca_dataframe = ApplyPCA(X_train_noLabel)
pca_dataframe_test= pcaModel.transform(X_test_noLabel)

print("PCA CLASSIFICATION")

logreg_pca = LogisticRegression(dual=False,class_weight='balanced').fit(pca_dataframe,y_train)
#logreg_pca = LogisticRegression().fit(pca_dataframe,y_train)

y_predective_log_pca= logreg_pca.predict(pca_dataframe_test)
calculateStatsModel(y_test,y_predective_log_pca)


#loadingPCA(pcaModel,pca_dataframe,X_train_noLabel,X_train_scored)

print("EXIT...")

