"""
@author: Linh Van NGUYEN
Date: 16/5/2016
"""

import scipy.sparse
import utils as utils
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")

def predict_singlemodel(tweetsTrain, labelsTrain, tweetsTest):
    # Function to predict the sentiment (0,1,2 as positive, neutral and negative)
    # using the best single model: linear SVM with significant stacked features 
    # chosen by a Random Forest classifier
    #
    nTrain = len(tweetsTrain)   
    # nTest = len(tweetsTest)   

    # 1. Clean text ===========================================================
    print("Step 1: Cleaning text \n")
    texts = []
    for tweet in tweetsTrain:
        texts.append(" ".join(utils.text_to_wordlist(tweet, False)))
    for tweet in tweetsTest:
        texts.append(" ".join(utils.text_to_wordlist(tweet, False)))
    
    # 2. Extract features ===================================================== 
    print("Step 2: Extracting features \n")
    maxFeatures = 2000
    vectorizer = CountVectorizer(
        analyzer = 'word',
        tokenizer = None,
        lowercase = True,
        stop_words = None,
        max_features = maxFeatures
    )
        
    tfv = TfidfVectorizer(min_df=3,  max_features=maxFeatures, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
    
    feasVect = tfv.fit_transform(texts)
    feasCountVect = vectorizer.fit_transform(texts)  
    # feasStacked = scipy.sparse.hstack([feasVect, feasCountVect])
    # Train and Test
    feasVectTrain = feasVect[:nTrain]
    feasVectTest = feasVect[nTrain:]
    feasCountVectTrain = feasCountVect[:nTrain]
    feasCountVectTest = feasCountVect[nTrain:]
    feasStackedTrain =  scipy.sparse.hstack([feasVectTrain, feasCountVectTrain])
    feasStackedTest = scipy.sparse.hstack([feasVectTest, feasCountVectTest])

    #3: Train classifiers ===================================================== 
    print("Step 3: Training classifiers \n")

    
    print("stacked features ....")
    clfRF = RandomForestClassifier(n_estimators=250)
    clfRFStacked = clfRF.fit(feasStackedTrain, labelsTrain)
    clfLR = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, 
                               intercept_scaling=1.0, class_weight=None, random_state=None)

    selModel = SelectFromModel(clfRFStacked, "1.0*mean", prefit=True)
    feasStackedTrainSel = selModel.transform(feasStackedTrain)
    feasStackedTestSel = selModel.transform(feasStackedTest)

    clfLRStackedSel = clfLR.fit(feasStackedTrainSel, labelsTrain)
    labelsPredTestLRStackedSel = clfLRStackedSel.predict(feasStackedTestSel)
    return labelsPredTestLRStackedSel
    print("done! \n")

def predict_ensemblemodels(tweetsTrain, labelsTrain, tweetsTest):

    nTrain = len(tweetsTrain)   
    # nTest = len(tweetsTest)   

    # 1. Clean text ===========================================================
    print("Step 1: Cleaning text \n")
    texts = []
    for tweet in tweetsTrain:
        texts.append(" ".join(utils.text_to_wordlist(tweet, False)))
    for tweet in tweetsTest:
        texts.append(" ".join(utils.text_to_wordlist(tweet, False)))
    
    # 2. Extract features ===================================================== 
    print("Step 2: Extracting features \n")
    maxFeatures = 2000
    vectorizer = CountVectorizer(
        analyzer = 'word',
        tokenizer = None,
        lowercase = True,
        stop_words = None,
        max_features = maxFeatures
    )
        
    tfv = TfidfVectorizer(min_df=3,  max_features=maxFeatures, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
    
    feasVect = tfv.fit_transform(texts)
    feasCountVect = vectorizer.fit_transform(texts)  
    # feasStacked = scipy.sparse.hstack([feasVect, feasCountVect])
    # Train and Test
    feasVectTrain = feasVect[:nTrain]
    feasVectTest = feasVect[nTrain:]
    feasCountVectTrain = feasCountVect[:nTrain]
    feasCountVectTest = feasCountVect[nTrain:]
    feasStackedTrain =  scipy.sparse.hstack([feasVectTrain, feasCountVectTrain])
    feasStackedTest = scipy.sparse.hstack([feasVectTest, feasCountVectTest])

    #3: Train classifiers ===================================================== 
    print("Step 3: Training classifiers \n")

    print("For vectorized features ....")
    clfLR = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, 
                               intercept_scaling=1.0, class_weight=None, random_state=None)
    clfRF = RandomForestClassifier(n_estimators=250)
    clfEXT = ExtraTreesClassifier(n_estimators=300)
    clfADA = AdaBoostClassifier(n_estimators=100, learning_rate = 0.75)
    clfXGB = xgb.XGBClassifier(objective= 'logistic', nthread=3, silent = 1, seed=8,
                               n_estimators = 400, max_depth = 6, learning_rate = 0.075, 
                               subsample = 0.9, colsample_bytree = 1.0)    
    clfLRVect = clfLR.fit(feasVectTrain, labelsTrain)
    clfRFVect = clfRF.fit(feasVectTrain, labelsTrain)
    clfEXTVect = clfEXT.fit(feasVectTrain, labelsTrain)
    clfADAVect = clfADA.fit(feasVectTrain, labelsTrain)
    clfXGBVect = clfXGB.fit(feasVectTrain, labelsTrain)                               
    
    print("counting features ....")
    clfLR = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, 
                               intercept_scaling=1.0, class_weight=None, random_state=None)
    clfRF = RandomForestClassifier(n_estimators=250)
    clfEXT = ExtraTreesClassifier(n_estimators=300)
    clfADA = AdaBoostClassifier(n_estimators=100, learning_rate = 0.75)
    clfXGB = xgb.XGBClassifier(objective= 'logistic', nthread=3, silent = 1, seed=8,
                               n_estimators = 400, max_depth = 6, learning_rate = 0.075, 
                               subsample = 0.9, colsample_bytree = 1.0)       
    clfLRCountVect = clfLR.fit(feasCountVectTrain, labelsTrain)
    clfRFCountVect = clfRF.fit(feasCountVectTrain, labelsTrain)
    clfEXTCountVect = clfEXT.fit(feasCountVectTrain, labelsTrain)
    clfADACountVect = clfADA.fit(feasCountVectTrain, labelsTrain)
    clfXGBCountVect = clfXGB.fit(feasCountVectTrain, labelsTrain)
    
    print("stacked features ....")
    clfLR = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, 
                               intercept_scaling=1.0, class_weight=None, random_state=None)
    clfRF = RandomForestClassifier(n_estimators=250)
    clfEXT = ExtraTreesClassifier(n_estimators=300)
    clfADA = AdaBoostClassifier(n_estimators=100, learning_rate = 0.75)
    clfXGB = xgb.XGBClassifier(objective= 'logistic', nthread=3, silent = 1, seed=8,
                               n_estimators = 400, max_depth = 6, learning_rate = 0.075, 
                               subsample = 0.9, colsample_bytree = 1.0)   
    clfLRStacked = clfLR.fit(feasStackedTrain, labelsTrain)
    clfRFStacked = clfRF.fit(feasStackedTrain, labelsTrain)
    clfEXTStacked = clfEXT.fit(feasStackedTrain, labelsTrain)
    clfADAStacked = clfADA.fit(feasStackedTrain, labelsTrain)
    clfXGBStacked = clfXGB.fit(feasStackedTrain, labelsTrain)    
    
    print("and significant features ....")
    clfLR = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, 
                               intercept_scaling=1.0, class_weight=None, random_state=None)

    selModel = SelectFromModel(clfRFStacked, "1.0*mean", prefit=True)
    feasStackedTrainSel = selModel.transform(feasStackedTrain)
    feasStackedTestSel = selModel.transform(feasStackedTest)

    clfLRStackedSel = clfLR.fit(feasStackedTrainSel, labelsTrain)

    print("done! \n")
    
    #4: Model ensemble ========================================================
    print("Final predictions and ensemble \n")

    labelsPredTestLRVect = clfLRVect.predict_proba(feasVectTest)
    labelsPredTestRFVect = clfRFVect.predict_proba(feasVectTest)
    labelsPredTestEXTVect = clfEXTVect.predict_proba(feasVectTest)
    labelsPredTestADAVect = clfADAVect.predict_proba(feasVectTest)
    labelsPredTestXGBVect = clfXGBVect.predict_proba(feasVectTest)

    labelsPredTestLRCountVect = clfLRCountVect.predict_proba(feasCountVectTest)
    labelsPredTestRFCountVect = clfRFCountVect.predict_proba(feasCountVectTest)
    labelsPredTestEXTCountVect = clfEXTCountVect.predict_proba(feasCountVectTest)
    labelsPredTestADACountVect = clfADACountVect.predict_proba(feasCountVectTest)
    labelsPredTestXGBCountVect = clfXGBCountVect.predict_proba(feasCountVectTest)   
    
    labelsPredTestLRStacked = clfLRStacked.predict_proba(feasStackedTest)
    labelsPredTestRFStacked = clfRFStacked.predict_proba(feasStackedTest)
    labelsPredTestEXTStacked = clfEXTStacked.predict_proba(feasStackedTest)
    labelsPredTestADAStacked = clfADAStacked.predict_proba(feasStackedTest)
    labelsPredTestXGBStacked = clfXGBStacked.predict_proba(feasStackedTest)    
    
    labelsPredTestLRStackedSel = clfLRStackedSel.predict_proba(feasStackedTestSel)
    
    probsPred  = (labelsPredTestLRVect+labelsPredTestRFVect+labelsPredTestEXTVect+ 
                 labelsPredTestADAVect+ labelsPredTestXGBVect+
                 labelsPredTestLRCountVect+labelsPredTestRFCountVect+labelsPredTestEXTCountVect+
                 labelsPredTestADACountVect+labelsPredTestXGBCountVect+
                 labelsPredTestLRStacked+labelsPredTestRFStacked+labelsPredTestEXTStacked+
                 labelsPredTestADAStacked+labelsPredTestXGBStacked+
                 labelsPredTestLRStackedSel)/16.0
    
    # choosing the label as the one with maximum probability
    labelsPred = np.argmax(probsPred, axis=1)             
    return labelsPred    
    print("Done ensembling \n")