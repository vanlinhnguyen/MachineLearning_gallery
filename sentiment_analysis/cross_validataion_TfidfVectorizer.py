"""
@author: Linh Van NGUYEN
Date: 16/5/2016
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import utils as utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import cross_validation
from sklearn import grid_search
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import xgboost as xgb

   
# Load training text
filename = 'train-tweets.txt'
all_text = utils.read_textfile(filename,',')
df = pd.DataFrame(all_text,columns=['sentiment', 'tweets'])
tweets = df.tweets.values
labels = pd.factorize(df.sentiment)[0]
sentiments = pd.factorize(df.sentiment)[1]


skf_cv = cross_validation.StratifiedKFold(labels, n_folds=6, shuffle=True, random_state=None) # split target equally

print "Cleaning and parsing tweets...\n"      
features = []
for tweet in tweets:
    features.append(" ".join(utils.text_to_wordlist(tweet, False)))

print 'Vectorizing... \n', 
# http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
tfv = TfidfVectorizer(min_df=3,  max_features=2000, 
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words = 'english')


features = tfv.fit_transform(features)


clfLR = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, 
                           intercept_scaling=1.0, class_weight=None, random_state=None)

clfNB = MultinomialNB()
clfSGD = SGDClassifier(loss='hinge', penalty='l2', alpha = 3*1e-4, n_iter=5, random_state=42)
clfRF = RandomForestClassifier(n_estimators=250)
clfEXT = ExtraTreesClassifier(n_estimators=300)
clfADA = AdaBoostClassifier(n_estimators=100, learning_rate = 0.75)
clfXGB = xgb.XGBClassifier(objective= 'logistic', nthread=3, silent = 1, seed=8,
                           n_estimators = 500, max_depth = 6, learning_rate = 0.075, 
                           subsample = 0.9, colsample_bytree = 1.0)

print("Accuracy ( LR):", np.mean(utils.score_model(clfLR ,features,labels, cv = skf_cv, scoring = "accuracy")))
print("Accuracy ( NB):", np.mean(utils.score_model(clfNB ,features,labels, cv = skf_cv,scoring = "accuracy")))
print("Accuracy (SVM):", np.mean(utils.score_model(clfSGD,features,labels, cv = skf_cv,scoring = "accuracy")))
print("Accuracy ( RF):", np.mean(utils.score_model(clfRF ,features,labels, cv = skf_cv,scoring = "accuracy")))
print("Accuracy (EXT):", np.mean(utils.score_model(clfEXT,features,labels, cv = skf_cv,scoring = "accuracy")))
print("Accuracy (ADA):", np.mean(utils.score_model(clfADA,features,labels, cv = skf_cv,scoring = "accuracy")))
print("Accuracy (XGB):", np.mean(utils.score_model(clfXGB,features,labels, cv = skf_cv,scoring = "accuracy")))


####### SGD ##########
"""
paramsSGD = {
    'alpha':[1e-5, 1e-4, 1e-3, 1e-2] # best: 1e-4
}
"""
paramsSGD = {
    'alpha':[5* 1e-5, 1e-4, 1e-4, 2*1e-4, 3*1e-4, 4*1e-4, 5*1e-4] # best: 3*1e-4
}

clfSGD = SGDClassifier(loss='hinge', penalty='l2', n_iter=5, random_state=42)
clfGridSGD = grid_search.GridSearchCV(clfSGD, paramsSGD, n_jobs=3, 
                                      cv=skf_cv, scoring='accuracy', verbose = 1)
# clfGridSGD.fit(features, labels)
# print clfGridSGD.best_params_

####### RF ##########
"""
paramsRF = {
    'n_estimators':[10, 50, 100, 500, 1000] # best: 500
}

paramsRF = {
    'n_estimators':[200, 350, 500, 650, 800] # best: 350
}

paramsRF = {
    'n_estimators':[250, 300, 350, 400, 450] # best: 250
}

"""
paramsRF = {
    'n_estimators':[200, 225, 250, 275, 300]
}

clfRF = RandomForestClassifier()
clfGridRF = grid_search.GridSearchCV(clfRF, paramsRF, n_jobs=3, 
                                      cv=skf_cv, scoring='accuracy', verbose = 1)
#clfGridRF.fit(features, labels)
#print clfGridRF.best_params_

####### EXT ##########
"""
paramsEXT = {
    'n_estimators':[10, 50, 100, 500, 1000] # best: 500
}

paramsEXT = {
    'n_estimators':[200, 350, 500, 650, 800] # best: 350
}

paramsEXT = {
    'n_estimators':[250, 300, 350, 400, 450] # best 300
}

"""

paramsEXT = {
    'n_estimators':[250, 300, 350, 400, 450]
}

clfEXT = ExtraTreesClassifier()
clfGridEXT = grid_search.GridSearchCV(clfEXT, paramsEXT, n_jobs=3, 
                                      cv=skf_cv, scoring='accuracy', verbose = 1)
# clfGridEXT.fit(features, labels)
# print clfGridEXT.best_params_

####### ADA ##########
"""
paramsADA = {
    'n_estimators':[10, 50, 100, 500, 1000],
    'learning_rate':[0.01, 0.02, 0.1, 0.5, 1] # best: 100, 0.5
}
paramsADA = {
    'n_estimators':[50, 100, 250, 500],
    'learning_rate':[0.25, 0.5, 0.75, 1] # best: 100, 0.5
}


"""
paramsADA = {
    'n_estimators':[50, 75, 100, 125, 150],
    'learning_rate':[0.5, 0.75, 1]
}

clfADA = AdaBoostClassifier()
clfGridADA = grid_search.GridSearchCV(clfADA, paramsADA, n_jobs=3, 
                                      cv=skf_cv, scoring='accuracy', verbose = 1)
#clfGridADA.fit(features, labels)
#print clfGridADA.best_params_

####### XGB ##########

"""
paramsXGB = {
    'n_estimators':[50, 100, 300, 500], # 500
    'max_depth': [2, 3, 5], # 5
    'learning_rate': [0.05, 0.1, 0.2, 0,5], # 0.1
    'subsample': [0.9, 1.0], # 0.9
    'colsample_bytree': [0.9, 1.0] # 1
}

paramsXGB = {
    'n_estimators':[500, 750], # 500
    'max_depth': [5, 7], # 7
    'learning_rate': [0.075, 0.1, 0.2], # 0.075
    'subsample': [0.9], # 0.9
    'colsample_bytree': [1.0] # 1
}

paramsXGB = {
    'n_estimators':[400, 500, 600], # 400
    'max_depth': [5, 7, 9], # 5
    'learning_rate': [0,05, 0.075, 0.1], # 0.075
    'subsample': [0.9], # 0.9
    'colsample_bytree': [1.0] # 1
}

paramsXGB = {
    'n_estimators':[300, 400, 500], # 500
    'max_depth': [4, 5, 6], # 6
    'learning_rate': [0.075], # 0.075
    'subsample': [0.9], # 0.9
    'colsample_bytree': [1.0] # 1
}

"""
paramsXGB = {
    'n_estimators':[400, 500, 600], # 500
    'max_depth': [5, 6, 7], # 6
    'learning_rate': [0.075], # 0.075
    'subsample': [0.9], # 0.9
    'colsample_bytree': [1.0] # 1
}

clfXGB = xgb.XGBClassifier(
      objective= 'logistic', 
      nthread=3,
      silent = 1,
      seed=8
)

clfGridXGB = grid_search.GridSearchCV(clfXGB, paramsXGB, n_jobs=3, 
                                      cv=skf_cv, scoring='accuracy', verbose = 1)
#clfGridXGB.fit(features, labels)
#print clfGridXGB.best_params_
