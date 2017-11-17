# -*- coding: utf-8 -*-
"""
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

#specify path for nltk files
jar = 'C:/Users/Cedric/Workspace_Python/Hackaton/Hackaton_Xtramile/stanford-postagger-full-2017-06-09/stanford-postagger-3.8.0.jar'
modelTager = 'C:/Users/Cedric/Workspace_Python/Hackaton/Hackaton_Xtramile/stanford-postagger-full-2017-06-09/models/french.tagger'

java_path = "C:/ProgramData/Oracle/Java/javapath/java.exe"
os.environ['JAVAHOME'] = java_path


#create the pos_tagger and the stemmer
pos_tagger = StanfordPOSTagger(modelTager, jar, encoding='utf8' )
#res = pos_tagger.tag('je suis libre de prendre un bus'.split())
#print (res)

stemmer = SnowballStemmer("french")
#stemmer.stem("prendre")


#Implements dunction to clean description and keywords

#function to delete non importnt word
def finalText1(init_text):
    
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
            if (len(word)>3) and (word not in final_edited_sentence):
                final_edited_sentence.append(word)
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






dataset = pd.read_csv("C:/Users/Cedric/Workspace_Python/Hackaton/Hackaton_Xtramile/datasetjob.csv", sep=";",decimal=".")
dataset=dataset[dataset["description"].isnull()==False] #we drop the row where the description is null
#drop row 20 because there is an nan for the description, just use in the case of this dataset
#df = df.drop(20,axis = 0)
dataset = dataset.reset_index()
dataset = dataset.drop("index",axis = 1)#when we reindex we got 2 index columns so we delete one

#Apply differents function to clean keywords and description
df_total=dataset[["description","keywords"]]
df_total["keywords"]=df_total["keywords"].apply(keyword) #We clean keywords text
df_total["description"] = df_total["description"].apply(fromHtlmToText) #delete html format from description. Make it readable
df_total["description"]=df_total["description"].apply(finalText) #we clean descriptions(deleting stopwords)




#We create function Transformer to take a part of total df
get_description = FunctionTransformer(lambda x: x['description'], validate=False)
get_keywords = FunctionTransformer(lambda x: x['keywords'], validate=False)

                



#TF-IDF for description
tfidf_vectorizer=TfidfVectorizer(analyzer='word',ngram_range=(1,1),strip_accents="unicode")

#Bag of word for keywords
bagOfWord_vectorizer=CountVectorizer(analyzer='word',ngram_range=(1,1),strip_accents="unicode")

######   BELOW IS NOT IMPORTANT FOR TESTING
#mod_res=tfidf_vectorizer.fit(df_total["description"])
#matrice_mod=mod_res.transform(df_total["description"]).toarray()
#
#def count_numbWord(matrice):
#    tab_nombre=[]
#    for i in range(0,matrice.shape[1]):
#        nombre_non_nuls=np.sum(matrice[:,i]!=0)
#        tab_nombre.append(nombre_non_nuls)
#    return(tab_nombre)
#colonne_nonNuls=count_numbWord(matrice_mod)
#dataFrame_mots=pd.DataFrame(np.column_stack([mod_res.get_feature_names(),colonne_nonNuls])
#                           ,columns=["Mots","nombre_doc"])    
##### ABOVE IS NOT IMPORTANT FOR TESTING


#A feature Union to join countVectorize on keyWords and tf-idf on Description
process_and_join_features = FeatureUnion(
            transformer_list = [
                ('description_feature', Pipeline([
                    ('selector', get_description),
                    ('tf_idf', tfidf_vectorizer)
                ])),
                ('keywords_feature', Pipeline([
                    ('selector', get_keywords),
                    ("bagOfWord", CountVectorizer ()), #ne pas mettre d'arg a CountVectorizer revient a tokeniser par "space"
                    
              ]))
             ]
        )
                
# Instantiate nested pipeline: pl
pipeline = Pipeline([
        ('union', process_and_join_features),
        ('scaler',StandardScaler(with_mean=False)) 
    ])



pipeline.fit(df_total)
matrice=pipeline.fit_transform(df_total).todense()

#instanciate a Kmeans and then compute a silhouette_score
kmeanModel_example=KMeans(n_clusters=16).fit(matrice)
model_example=kmeanModel_example.predict(matrice)
silhouette_score(matrice,kmeanModel_example.labels_)


###This part was to find words which come the most in documents "les mots qui reviennent dans le plus de documents"
#def findWholeWord(w):
#    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search
#nombres_occurences=[]
#for mot in vector.get_feature_names():
#    nombre=0
#    print(mot)
#    for i in range(0,df.shape[0]-1):
#        document=concat_df[i]
#        if findWholeWord(mot)(document) != None :
#            nombre=nombre+1
#    nombres_occurences.append(nombre)
#ind=nombres_occurences.index(max(nombres_occurences))
#vector.get_feature_names()[ind]

#dataFrame_mots=pd.DataFrame(np.column_stack([vector.get_feature_names(),nombres_occurences])
#                            ,columns=["Mots","nombre_occurences"])  

 
#The elbow Method to find best k
distortions = []
K = range(40,60)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(matrice)
    kmeanModel.fit_predict(matrice)
    distortions.append(sum(np.min(cdist(matrice, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / matrice.shape[0])
  
# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
