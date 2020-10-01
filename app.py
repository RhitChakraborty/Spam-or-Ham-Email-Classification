# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 01:48:27 2020

@author: rhitc
"""

import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

def clean_text(text):
    """
    Parameters
    ----------
    text : TYPE: String
        DESCRIPTION: Clean texts, remove Stopwords and stemming

    Returns: A string of cleaned words
    -------
    """
    text=re.sub('[^a-zA-Z0-9]+',' ',text).lower().split()
    ps=PorterStemmer()
    text=' '.join([ps.stem(w) for w in text if w not in set(stopwords.words('english'))])
    
    return text

cv=pickle.load(open('models/cv.pkl','rb'))
model=pickle.load(open('models/final_clf.pkl','rb'))


st.title('Email Spam/Ham Detector')
text=st.text_input('Enter your email message here :')
text=clean_text(text)
text=cv.transform([text])
pred=model.predict(text)[0]

st.write('Click predict to know your answer')
if st.button('Predict'):
    if pred==1:
        st.error('SPAM Alert!!')
    else:
        st.success('NOT SPAM!! This may be important to you')

st.text('Author:RhitC')
