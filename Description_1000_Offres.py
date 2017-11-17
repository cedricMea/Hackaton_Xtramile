# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 10:23:35 2017

@author: Cedric
"""
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import urllib.request
from bs4 import BeautifulSoup
import sys
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from nltk.stem import SnowballStemmer
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from nltk.tag import StanfordPOSTagger
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from langdetect import detect
from sklearn.preprocessing import FunctionTransformer,StandardScaler,MaxAbsScaler
from sklearn.pipeline import FeatureUnion,Pipeline
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram,linkage



#specify path for nltk files
jar = 'C:/Users/Cedric/Workspace_Python/Hackaton/Hackaton_Xtramile/stanford-postagger-full-2017-06-09/stanford-postagger-3.8.0.jar'
modelTager = 'C:/Users/Cedric/Workspace_Python/Hackaton/Hackaton_Xtramile/stanford-postagger-full-2017-06-09/models/french.tagger'

java_path = "C:/ProgramData/Oracle/Java/javapath/java.exe"
os.environ['JAVAHOME'] = java_path


#create the pos_tagger and the stemmer
pos_tagger = StanfordPOSTagger(modelTager, jar, encoding='utf8' )

#create the stemmer
stemmer = SnowballStemmer("french")



#Implements dunction to clean description and keywords
#function to delete non importnt word
def finalText(init_text):
    
    text1=init_text.lower()
    text2=re.sub(r"[\s(,|;|.|:|//)]"," ",text1)
    text2=text2.replace("'"," ")
    text=text2.replace("’"," ")
    #text=text2.translate(str.maketrans({"'":" "}))
    #print(text2)
    list_word=text.split(" ")
    english_stopwords = set(STOPWORDS)
    #detect the language
    number_en_stop_word=0
    english=False
    for word in english_stopwords:
        if word in list_word:
            number_en_stop_word=number_en_stop_word+1
        if (number_en_stop_word/len(list_word)>0.10):
            english=True
            break
    if english:
        list_word=text.split(" ")
        resultwords  = [word for word in list_word if word.lower() not in english_stopwords]
        final_text=' '.join(resultwords)
    else:
        #We have a text in french
        #♦print("etape 3")
        pos_tagger = StanfordPOSTagger(modelTager, jar, encoding='utf8' )
        res = pos_tagger.tag(list_word)
        #print("etape4")
        #[print(word,tag) for word,tag in res]
        edited_sentence = [word for word,tag in res if tag != 'V' and tag != 'NNPS' and tag!='DET' and tag!='P' and tag!='CLS' and tag !='CLO' and tag!='VIMP' and tag!='CC' and tag !='P'
                           and tag != 'PRO' and tag != 'PROREL' and tag != 'ADV']
        #final_edited_sentence= [stemmer.stem(word) for word in edited_sentence if len(word)>3 ]
        final_edited_sentence=[]
        for word in edited_sentence :
            if (len(word)>3) and (stemmer.stem(word) not in final_edited_sentence):
                final_edited_sentence.append(stemmer.stem(word))
        #print(pos_tagger.tag(edited_sentence))
        final_text=' '.join(final_edited_sentence)
    
   
    return (final_text)


# to convert the description into a readable text
def fromHtlmToText(textHtml):
    tampon = BeautifulSoup(textHtml,'html.parser')
    Stringtampon = tampon.get_text()
    Stringtampon = Stringtampon.replace("\n"," ")

    Stringtampon = Stringtampon.replace("\xa0 "," ")
    Stringtampon = Stringtampon.replace("\xa0"," ")
    return (Stringtampon)


#function to clean keyword
def keyword(key):
    clean_key=re.sub(r'(\{|\}|\\|"|\|:|\')',"",key)
    clean_key=re.sub(r'(\s(ans))',"_ans",clean_key)#On traduit 5 ans en 5_ans
    clean_key=clean_key.replace(","," ")
    list_words=clean_key.split(" ")
    #steam_word=[stemmer.stem(word) for word in list_words if  stemmer.stem(word) not in steam_word]
    steam_word=[]
    for word in list_words :
            if (stemmer.stem(word) not in steam_word):
                steam_word.append(stemmer.stem(word))

    return(' '.join(steam_word))



# read all the jobs in the folder
def mylistdir(directory):

    filelist = os.listdir(directory)
    print(filelist)
    return [x for x in filelist
            if not (x.startswith('.'))]


dataFrame=pd.Series() #the dataframe of modified job_description
dataFrame_all=pd.Series() #dataframe of original job_description
folder_path = "C:/Users/Cedric/Desktop/offers-mix/offers-mix"
test= mylistdir(folder_path)


debut=0 #debut and fin is to allow to take 1000 offers by many interval. From debut to fin each time
fin=1000
ajout=pd.Series() #Ajout is a Series which is like a tem for each time i want to add new job_description
for index,element in enumerate(test):
     if index>=debut and index<fin :
         pathname =folder_path +"/"+element 
         abs_path=re.sub(r"(\\)","/",pathname) #I am on Windows so I have to  replace "\" by "/"
         a=open(str(abs_path),encoding="utf-8")
         b=a.read() 
         ajout.loc[index-debut]=b
     
for indice in range(0,ajout.shape[0]):
    dataFrame.loc[debut+indice]=finalText(ajout[indice]) #I add modified job_description to dataFrame
    dataFrame_all.loc[debut+indice]=ajout[indice] #I add original job_description to dataFrame_all
    #print(debut+indice) it 's just to see the progress of the filling of dataFrame
    
    


#Construction of tf-idf matix
tfidf_vectorizer=TfidfVectorizer(analyzer='word',ngram_range=(1,1),strip_accents="unicode")
mod_res=tfidf_vectorizer.fit(dataFrame)
matrice_mod=mod_res.transform(dataFrame).todense()


#Construction of KMean Model
kmeanModel = KMeans(n_clusters=200).fit(matrice_mod)
res=kmeanModel.predict(matrice_mod)


#Construct  and plot different intra_cluster distortions and silhouette_score for different score
distortions = []
silhouette=[]
K =[50,100,150,200,250,300]
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(matrice_mod)
    kmeanModel.fit_predict(matrice_mod)
    #distortions.append(sum(np.min(cdist(matrice_mod, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / matrice_mod.shape[0])
    #silhouette.append(silhouette_score(matrice_mod,kmeanModel.labels_))   
#Plot the elbow
plt.plot(K, silhouette, 'bx-')
plt.xlabel('k')
plt.ylabel('Silhouette_score')
plt.title('The silhouette Method showing the optimal k')
plt.show()


#Construction and plot of a Dendogram 
Z=linkage(matrice_mod,method="ward")
fig = plt.figure(1, figsize=(20, 10))    
#○dendogram
dendrogram(Z,p=200,color_threshold =2.5)
plt.title("Dendrogram")
plt.xlabel("Customer")
plt.ylabel("Euclidean distances")
plt.show()
