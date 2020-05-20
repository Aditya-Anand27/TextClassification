#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os


# In[2]:


os.chdir(r"C:\Users\balad\Downloads\bbc-full-text-document-classification\bbc-fulltext (document classification)\bbc")
traindata=[]
trainlabel=[]
for i in ["business","sport","tech","politics","entertainment"]:
    contents=os.listdir(i)
    for j in contents:
        file=i+'/'+j
        if(os.stat(file).st_size>0):
            with open(file) as openfile:
                text=openfile.readlines()
            text=" ".join(text)
            traindata.append(text)
            trainlabel.append(i)


# In[3]:


print(traindata[-4:-1])
print(trainlabel[-4:-1])


# In[4]:


get_ipython().system('pip install nltk')
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score


# In[5]:


import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')


# In[6]:


print(traindata[0])


# In[7]:


from keras.preprocessing.text import text_to_word_sequence
for i in range(0,len(traindata)):
    traindata[i]=text_to_word_sequence(traindata[i])
    


# In[8]:


print(traindata[0:2])


# In[51]:


from sklearn.feature_extraction.text import TfidfVectorizer

def retdoc(doc):
    return doc

tfidf = TfidfVectorizer(
    analyzer='word',
    tokenizer=retdoc,
    preprocessor=retdoc,
    token_pattern=None)
tfidf.fit(traindata)


# In[10]:


print(tfidf.vocabulary_)


# In[52]:


traindatatoken = tfidf.transform(traindata)
Naive = naive_bayes.MultinomialNB()
Naive.fit(traindatatoken,trainlabel)


# In[53]:


testdata=[]
with open(r"C:\Users\balad\Downloads\classify_text.txt",encoding="utf8") as openfile:
    while(openfile.readline()):
        text=openfile.readline()
        if(len(text)!=1):
            testdata.append(str(text))
print(testdata[8])


# In[54]:


from keras.preprocessing.text import text_to_word_sequence
testdatatoken=[]
for i in range(0,len(testdata)):
    testdatatoken.append(text_to_word_sequence(testdata[i]))


# In[55]:


print(testdatatoken)


# In[56]:


testdatacheck = tfidf.transform(testdatatoken)
predict = Naive.predict(testdatacheck)


# In[57]:


print(predict)


# In[61]:


for i in range(0,len(testdata)):
    print("The phrase(as tokens) is ")
    print(testdatatoken[i])
    print("The prediction is ")
    print(predict[i])
    print("")


# In[62]:


predictions_NB = Naive.predict(tfidf.transform([['Trump','instructs','business','to','work','from','home']]))
print(predictions_NB)


# In[ ]:




