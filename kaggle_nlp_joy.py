#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 23:37:03 2019

@author: thisisjoy
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns

import re
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer



#======================================#

#             Data Insight             #

#======================================#

# Importing the dataset:
yelp = pd.read_csv('kaggle_dataset/yelp_10k.csv')

# Create a new column where we can see the text length of each review:
yelp['text length'] = yelp['text'].apply(len)

# Let’s group the data by the star rating, and see if we can find a correlation 
# between features such as cool, useful, and funny. 
stars = yelp.groupby('stars').mean()
stars.corr()
# funny is strongly correlated with useful, 
# and useful seems strongly correlated with text length.


# Our task is to predict if a review is either bad or good, 
# so let’s just grab reviews that are either 1 or 5 stars from the yelp dataframe:
yelp_class  = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]
# yelp_class.shape

X = yelp_class['text']
y = yelp_class['stars']




#======================================#

#          Data preprocessing          #

#======================================#

corpus = []
ps = PorterStemmer()

def text_process(text):
    '''
    Takes in a string of text, then performs the following:
    1. Remove all punctuation and numbers (only leave the words with letters a-zA-Z;)
    2. Remove all stopwords
    3. Convert into lower case
    4. Stemming
    5.. Return the cleaned text as a list of words
    '''
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english')) ]
    review = ' '.join(review)
    return review
    

for i in range(yelp_class.shape[0]):
    review = text_process(yelp_class['text'].iloc[i])
    corpus.append(review)
    

# Convert the text collection into a matrix of token counts,
# where each column is a unique word and each row is a review.
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()



#======================================#

#         Training Our Models          #

#======================================#


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# Multinomial Naive Bayes is a specialised version of Naive Bayes designed more for text documents.
from sklearn.naive_bayes import MultinomialNB, GaussianNB
mul_nb = MultinomialNB()
mul_nb.fit(X_train, y_train)
mul_y_pred = mul_nb.predict(X_test)


# Gaussian Naive Bayes
gaussian_nb = GaussianNB()
gaussian_nb.fit(X_train, y_train)
gaussian_y_pred = gaussian_nb.predict(X_test)


#======================================#

#           Model Evaluation           #

#======================================#

from sklearn.metrics import confusion_matrix, classification_report
mul_cm = confusion_matrix(y_test, mul_y_pred)
mul_report = classification_report(y_test, mul_y_pred)
print(mul_cm)
print('\n')
print(mul_report)


gaussian_cm = confusion_matrix(y_test, gaussian_y_pred)
gaussian_report = classification_report(y_test, gaussian_y_pred)
print(gaussian_cm)
print('\n')
print(gaussian_report)


# Multinomial can perform good when shrink the max_feature from 13000 to 1500 with avg acc: 92%


















