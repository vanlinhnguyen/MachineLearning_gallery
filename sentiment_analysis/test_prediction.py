"""
@author: Linh Van NGUYEN
Date: 16/5/2016
"""

import warnings
warnings.filterwarnings("ignore")
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

import utils as utils
import predict_sentiments as ps

fileTrain = 'train-tweets.txt'
fileTest = sys.argv[1] # get from terminal

dfTrain = utils.read_textfile(fileTrain,',')
tweetsTrain = dfTrain.tweet.values
nTrain = len(tweetsTrain)

dfTest = utils.read_textfile(fileTest,',')
tweetsTest = dfTest.tweet.values
nTest = len(tweetsTest)

sentimentsAll = np.concatenate((dfTrain.sentiment,dfTest.sentiment),axis=0) # to make sure the order when factorize
labelsAll = pd.factorize(sentimentsAll)[0]
sentiments = pd.factorize(dfTrain.sentiment)[1]
labelsTrain = labelsAll[:nTrain]
labelsTest = labelsAll[nTrain:]

print ("\n\nTest file at \" %s \", containing %d tweets \n\n" % (fileTest, nTest))

# Single model
print("==============================================")
print("Report of prediction by the best single models")
print("==============================================")
labelsPred = ps.predict_singlemodel(tweetsTrain, labelsTrain, tweetsTest)
print(classification_report(labelsPred, labelsTest))
print("==============================================")


# Ensemble model
print("==============================================")
print("Report of prediction by the best single models")
print("==============================================")
labelsPred = ps.predict_ensemblemodels(tweetsTrain, labelsTrain, tweetsTest)
print(classification_report(labelsPred, labelsTest))
print("==============================================")
