"""
@author: Linh Van NGUYEN
Date: 16/5/2016
"""

import re
from sklearn import cross_validation
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import pandas as pd

def read_textfile(filename, separate):
    # Function to read a text file and return a data frame of 2 fields: sentiment and tweets
    #
    all_text = []
    with open(filename, "rt") as f:
        for line in f:
            all_text.append(line.split(separate, 1)) # split at only the first delimiter
    df = pd.DataFrame(all_text,columns=['sentiment', 'tweet'])
    return df

def score_model(model,X,t,cv,scoring):
    # Function to estimate the score of a model from the training set
    # using cross validation
    #    
    return cross_validation.cross_val_score(model, X, t, cv=cv, scoring=scoring,n_jobs=-1)


def text_to_wordlist( text, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    text = BeautifulSoup(text, "lxml").get_text()

    # 2. Remove non-letters
    #text = re.sub("[^a-zA-Z]"," ", text)
    #text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ", text)
    text = re.sub("(@[A-Za-z]+)|([^A-Za-z \t])|(\w+:\/\/\S+)"," ", text)
    
    # 3. Convert words to lower case and split them
    words = text.lower().split()

    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    # 5. Return a list of words
    return(words)