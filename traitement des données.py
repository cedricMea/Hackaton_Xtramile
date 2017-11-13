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
from sklearn import feature_extraction

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
    print(i)
    fromHtlmToText(df["description"][i])


df["desc_text"] = df["description"].apply(fromHtlmToText)
df = df.drop("description",axis = 1)


#desc = urllib.request.urlopen("https://timac-agro.jobs.xtramile.io/technico-commercial-agricole-88-h-f")
#desc = df["description"][0]





