
import numpy as np
import pandas as pd
import statsmodels.api as sm 

from sklearn.linear_model import LinearRegression,LogisticRegression,RidgeCV
from sklearn.metrics import r2_score
from scipy.stats import zscore
from matplotlib import pyplot as plt
from sklearn.feature_selection import SequentialFeatureSelector
from utility.UtilityFunctions import ReadDataset
import pickle

def DoBackWardManual(trainingDataset):
    #Backward Forward

    
    #0.95 P-value so we have to discard pvalue > 0.05
    X= np.append(arr = np.ones((25010,1)).astype(int), values = X_train, axis = 1)

    #print(X)

    #we are taking all feature for now
    X_train_opt = X[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]].copy()
    regressor_OLS = sm.OLS(endog = y_train, exog = X_train_opt).fit()
    print(regressor_OLS.summary())

    #actually we have all features major of 0.05
    #removing the 3th col
    X_train_opt = X[:,[0, 1, 2, 4, 5, 6, 7, 8, 9, 10]]

    regressor_OLS = sm.OLS(endog = y_train, exog = X_train_opt).fit()
    print(regressor_OLS.summary())

    #removing the 6th col
    X_train_opt = X[:,[0, 1, 2, 4, 5, 6, 8, 9, 10]].copy()

    regressor_OLS = sm.OLS(endog = y_train, exog = X_train_opt).fit()
    print(regressor_OLS.summary())

    #removing the 2th col
    X_train_opt = X[:,[0, 1, 4, 5, 6, 8, 9, 10]].copy()
    regressor_OLS = sm.OLS(endog = y_train, exog = X_train_opt).fit()
    print(regressor_OLS.summary())

    #removing the 7th col
    X_train_opt = X[:,[0, 1, 4, 5, 6, 8, 9]].copy()
    regressor_OLS = sm.OLS(endog = y_train, exog = X_train_opt).fit()
    print(regressor_OLS.summary())

    #removing the 3th col
    X_train_opt = X[:,[0, 1, 4, 6, 8, 9]].copy()
    regressor_OLS = sm.OLS(endog = y_train, exog = X_train_opt).fit()
    print(regressor_OLS.summary())

    #removing the 2th col
    X_train_opt = X[:,[0, 1, 6, 8, 9]].copy()
    regressor_OLS = sm.OLS(endog = y_train, exog = X_train_opt).fit()
    print(regressor_OLS.summary())

    #removing the 4th col
    X_train_opt = X[:,[0, 1, 6, 8]].copy()
    regressor_OLS = sm.OLS(endog = y_train, exog = X_train_opt).fit()
    print(regressor_OLS.summary())

    #removing the 2th col
    X_train_opt = X[:,[0, 1, 8]].copy()
    regressor_OLS = sm.OLS(endog = y_train, exog = X_train_opt).fit()
    print(regressor_OLS.summary())

    #removing the 1th col
    X_train_opt = X[:,[0, 8]].copy()
    regressor_OLS = sm.OLS(endog = y_train, exog = X_train_opt).fit()
    print(regressor_OLS.summary())

    #R2 REMAINS 0, backward doesn't work!

def DoBackWardAutomatic(X_train,y_train):

    ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5)).fit(X_train, y_train)
    importance = np.abs(ridge.coef_)
    feature_names = np.array(X_train.columns)
    plt.bar(height=importance, x=feature_names)
    plt.title("Feature importances via coefficients")
    plt.show()

    plt.close()

    sfs_forward = SequentialFeatureSelector(ridge, n_features_to_select=4, direction="forward").fit(X_train, y_train)
    sfs_backward = SequentialFeatureSelector(ridge, n_features_to_select=4, direction="backward").fit(X_train, y_train)

    print("Features selected by forward sequential selection: "f"{feature_names[sfs_forward.get_support()]}")
    print("Features selected by backward sequential selection: "f"{feature_names[sfs_backward.get_support()]}")

    return feature_names[sfs_forward.get_support()]


#main

if __name__ == "__main__":
    trainingDataset = ReadDataset("./bin/resources/poker-hand-training-true.data")
    testingDataset = ReadDataset("./bin/resources/poker-hand-testing.data")

    y_train = trainingDataset['G']
    X_train = trainingDataset[['S1', 'R1','S2', 'R2','S3', 'R3','S4', 'R4','S5','R5']]

    y_test = testingDataset['G']
    X_test = testingDataset[['S1', 'R1','S2', 'R2','S3', 'R3','S4', 'R4','S5','R5']]
    

    with open('./bin/resources/testing_encodedDf.pickle', 'rb') as data:
        testingDataset_encoded = pickle.load(data)

    with open('./bin/resources/training-sampled_encodedDf.pickle', 'rb') as data:
        trainingDataset_sampled_encoded = pickle.load(data)

    #print(X_train)

    #Can we regress features? Mmmm
    # i.e Only Poker (7), we can regress the rank and the suit of one card.
    # Royal flush (9)
    #For others labels we cant.
    
    #Actually we introduce the dummy variable in order to reduce range for Suit and Rank.
    print("DUMMIES SUIT")
    X_train_dummy = pd.get_dummies(X_train, columns = ["S1","S2","S3", "S4", "S5"]).copy()
    print(X_train_dummy)

    #But is it necessary? Feautures increases to 25
    # isPicche, isCuori, isQuadri, isFiori

    # What about other features?


    #Feature selection for regression task
    #Every Card is revelant in order to achieve a Poker hand
    
    
    #DoBackWardManual(X_train)

    featuresSelected = DoBackWardAutomatic(X_train,y_train)

    X_train_selected = X_train[featuresSelected].copy()

    print(X_train_selected)
    
    #Linear
    regressorLinear = LinearRegression()
    regressorLinear.fit(X_train,y_train)

    y_predective_linear = regressorLinear.predict(X_test)
    score = r2_score(y_test,y_predective_linear)*100
    print("R2 Score original",score)


    regressorLinear.fit(X_train_selected,y_train)

    y_predective_linear = regressorLinear.predict(X_test[featuresSelected])
    score = r2_score(y_test,y_predective_linear)*100
    print("R2 Score selected",score)


    #Calculate LOG REG
    logreg = LogisticRegression(random_state=16,max_iter=10000)
    # fit the model with data
    logreg.fit(X_train,y_train)

    y_pred_reg = logreg.predict(X_test)
    score = r2_score(y_test,y_pred_reg)*100
    print("R2 Score log original",score)
    #Poor linear model, R2 is negative

    logreg.fit(X_train_selected,y_train)
    y_pred_reg = logreg.predict(X_test[featuresSelected])
    score = r2_score(y_test,y_pred_reg)*100
    print("R2 Score log selected",score)

    #We can try a costant regression model, but it will underfit everything

    #Maybe reduce varianze, thanks to zscore?
    print(X_train.var())
    X_train_scored = pd.DataFrame()
    X_test_scored = pd.DataFrame()
    for col in X_train.columns:
        X_train_scored[str(col)+'_zscore'] = zscore(X_train[col])
        X_test_scored[str(col)+'_zscore'] = zscore(X_test[col])

    #print(X_train_scored)

    #Variance after zcore
    print(X_train_scored.var())

    #Calculate LOG REG with scored dataset
    logreg = LogisticRegression(random_state=16,max_iter=1000)
    # fit the model with data
    logreg.fit(X_train_scored,y_train)

    y_pred_reg = logreg.predict(X_test_scored)
    score = r2_score(y_test,y_pred_reg)*100
    print("R2 Score scored log",score)



#try it https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html