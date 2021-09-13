# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def aucCV(features,labels):
    #models = [
        #RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
        #LinearSVC(),
        #MultinomialNB(),
        #SVC(kernel='linear',C=1.0),
        #LogisticRegression(random_state=0),
    param_test1 = {"n_estimators":range(1,200,10)}
    gsearch1 = GridSearchCV(estimator=RandomForestClassifier(),param_grid=param_test1,
                            scoring='roc_auc',cv=10)
    gsearch1.fit(features,labels)

    print(gsearch1.cv_results_)
    print(gsearch1.best_params_)
    print("best accuracy:%f" % gsearch1.best_score_)
    
    # return scores

def predictTest(trainFeatures,trainLabels,testFeatures):
    
    sel = VarianceThreshold(threshold=(.4 * (1 - .4)))
    std = StandardScaler()
    trainFeatures = std.fit_transform(trainFeatures)
    
    testFeatures = std.transform(testFeatures)
    

    model = RandomForestClassifier(n_estimators=161,max_features="log2", max_depth=10, random_state=10,oob_score=True)
    
    model.fit(trainFeatures, trainLabels)
    
    
    testOutputs = model.predict_proba(testFeatures)[:,1]
    
    
    return testOutputs
    
# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    data = np.loadtxt('spamTrain1.csv',delimiter=',')
    shuffleIndex = np.arange(np.shape(data)[0])
    np.random.shuffle(shuffleIndex)
    data = data[shuffleIndex,:]
    features = data[:,:-1]
    labels = data[:,-1]
    
    aucCV(features,labels)
    # print("10-fold cross-validation mean AUC: ",
    #       np.mean(aucCV(features,labels)))

    # X_train, X_test, y_train, y_test = \
    #     train_test_split(features, labels, test_size=.5, random_state=0)
    # testOutputs = predictTest(X_train,y_train,X_test)
    # trainFeatures = features[0::2,:]
    # trainLabels = labels[0::2]
    # testFeatures = features[1::2,:]
    # testLabels = labels[1::2]
    # testOutputs = predictTest(trainFeatures,trainLabels,testFeatures)
    # print("Test set AUC: ", roc_auc_score(testLabels,testOutputs))
    