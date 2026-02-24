import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as gbm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from skfeature.function.similarity_based import fisher_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from PLS import PLS
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
#from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import SequentialFeatureSelector as SFS
#import sklearn.feature_selection.SequentialFeatureSelector as SFS
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import re, os, sys
from collections import Counter
import math
def cv(clf, X, y, nr_fold):
    ix = []
    for i in range(0, len(y)):
        ix.append(i)
    ix = np.array(ix)

    allACC = []
    allSENS = []
    allSPEC = []
    allMCC = []
    allAUC = []
    for j in range(0, nr_fold):
        train_ix = ((ix % nr_fold) != j)
        test_ix = ((ix % nr_fold) == j)
        train_X, test_X = X[train_ix], X[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]
        clf.fit(train_X, train_y)
        p = clf.predict(test_X)
        pr = clf.predict_proba(test_X)[:,1]
        TP=0
        FP=0
        TN=0
        FN=0
        for i in range(0,len(test_y)):
            if test_y[i]==0 and p[i]==0:
                TP+= 1
            elif test_y[i]==0 and p[i]==1:
                FN+= 1
            elif test_y[i]==1 and p[i]==0:
                FP+= 1
            elif test_y[i]==1 and p[i]==1:
                TN+= 1
        ACC = (TP+TN)/(TP+FP+TN+FN)
        SENS = TP/(TP+FN)
        SPEC = TN/(TN+FP)
        det = math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        if (det == 0):
            MCC = 0
        else:
            MCC = ((TP*TN)-(FP*FN))/det
        AUC = roc_auc_score(test_y,pr)
        allACC.append(ACC)
        allSENS.append(SENS)
        allSPEC.append(SPEC)
        allMCC.append(MCC)
        allAUC.append(AUC)
    return np.mean(allACC),np.mean(allSENS),np.mean(allSPEC),np.mean(allMCC),np.mean(allAUC)

def test(clf, X, y, Xt, yt):
    train_X, test_X = X, Xt
    train_y, test_y = y, yt
    clf.fit(train_X, train_y)
    p = clf.predict(test_X)
    pr = clf.predict_proba(test_X)[:,1]
    TP=0
    FP=0
    TN=0
    FN=0
    for i in range(0,len(test_y)):
        if test_y[i]==0 and p[i]==0:
            TP+= 1
        elif test_y[i]==0 and p[i]==1:
            FN+= 1
        elif test_y[i]==1 and p[i]==0:
            FP+= 1
        elif test_y[i]==1 and p[i]==1:
            TN+= 1
    ACC = (TP+TN)/(TP+FP+TN+FN)
    SENS = TP/(TP+FN)
    SPEC = TN/(TN+FP)
    det = math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    if (det == 0):
        MCC = 0
    else:
        MCC = ((TP*TN)-(FP*FN))/det
    AUC = roc_auc_score(test_y,pr)

    return ACC, SENS, SPEC, MCC, AUC

TrainData=pd.read_csv('PF_CP_Feat.csv')
data=np.array(TrainData)
X=data[:,1:]
TestData=pd.read_csv('PF_CF_Feat_test_new.csv')
data_=np.array(TestData)
Xt=data_[:,1:]
Y1 = np.ones((620, 1))  # Value can be changed
Y2 = np.zeros((620, 1))
y = np.append(Y1, Y2)
Yt1 = np.ones((154, 1))  # Value can be changed
Yt2 = np.zeros((155, 1))
yt = np.append(Yt1, Yt2)
xgb_model=xgb.XGBClassifier()

xgbresult1=xgb_model.fit(X,y.ravel())
feature_importance=xgbresult1.feature_importances_
feature_number=-feature_importance
H1=np.argsort(feature_number)
mask=H1[:352]
train_data=X[:,mask]
test_tata=Xt[:,mask]


lr=LogisticRegression()
#xgb=XGBClassifier(n_estimators=param[i], learning_rate=0.1, random_state=0)
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


X = train_data
Xt = test_tata
# if set > X.shape[1]:
#     print(f"Skipping {set} features because it exceeds X.shape[1] = {X.shape[1]}")
#     continue

sfs = SFS(lr,
            k_features=set,
            forward=True,
            floating=False,
            scoring='accuracy',
            verbose=2,
            cv=5)

sfs = sfs.fit(X, y)

X_feature=sfs.transform(X)
Xt_feature=sfs.transform(Xt)
feat_cols = list(sfs.k_feature_idx_)

print(feat_cols)
#k_features = [len(k) for k in sfs.subsets_]
#plt.plot(k_features, sfs.scores_, marker='o')
#plt.ylim([0.7, 1.02])
#plt.ylabel('Accuracy')
#plt.xlabel('Number of features')
#plt.grid()
#plt.tight_layout()
#plt.show()
