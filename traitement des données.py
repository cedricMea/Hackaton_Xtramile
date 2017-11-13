# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 16:50:43 2017

@author: lulu
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 13:01:08 2017

@author: lulu
"""

import numpy as np
import pandas as pd
import nltk
import urllib.request
from bs4 import BeautifulSoup
import sys
from sklearn.feature_extraction.text import TfidfVectorizer

dataset = pd.read_csv("datasetjob.csv", sep=";",decimal=".")

df = dataset[["title","city","state","country","description","job_type","category"]]

#state has only null value
df = df.drop("state",axis = 1)


#drop row 20 because there is an nan for the description, just use in the case of this dataset
#df = df.drop(20,axis = 0)

#use in all case
df = df.dropna(subset=['description'], how='all')

df = df.reset_index()
df = df.drop("index",axis = 1)


def fromHtlmToText(textHtml):
    tampon = BeautifulSoup(textHtml,'html.parser')
    Stringtampon = tampon.get_text()
    Stringtampon = Stringtampon.replace("\n"," ")

    Stringtampon = Stringtampon.replace("\xa0 "," ")
    Stringtampon = Stringtampon.replace("\xa0"," ")
    return (Stringtampon)
    

#fromHtlmToText(df["description"][0])
    

for i in range(1,df.shape[0]):
    fromHtlmToText(df["description"][i])


df["desc_text"] = df["description"].apply(fromHtlmToText)
df = df.drop("description",axis = 1)


#desc = urllib.request.urlopen("https://timac-agro.jobs.xtramile.io/technico-commercial-agricole-88-h-f")
#desc = df["description"][0]


#CONCATENATION
concat_df=pd.Series(df.fillna(' ').values.tolist()).str.join(' ')
#print()
def indmax(M):
    ind = []
    max = 0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i][j] > max:
                max = M[i][j]
                
                


#Vectorizer
vector=TfidfVectorizer(analyzer='word',ngram_range=(4,4),strip_accents="unicode")
result=vector.fit_transform(concat_df)
#res1 = result.tocoo()
#k = res1.data.argmax()
#maxval = res1.data[k]
#maxrow = res1.row[k]
#maxcol = res1.col[k]

print (result.shape)
best_feature = result.argmax(axis = 1)
for i in range(1,len(best_feature)):
    vector.get_feature_names()[i][best_feature[i][0][1]]

vector.get_feature_names()[32]
concat_df[0]