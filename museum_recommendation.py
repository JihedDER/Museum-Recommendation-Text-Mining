#!/usr/bin/env python

'''
@ author : Jihed DEROUICHE

Subject: Museum Recommender engine (based on Museum description) [Based on content]

dataset= https://www.kaggle.com/annecool37/museum-data [Real data scrapped from TripAdvisor]

First Submission: This is my contribution in text similarity.( This code will be centralized on argument and executed as museum_recommender.py -i <customer Description>) 
First part of the code: TfidfVectorizer+ cosine_similarity
Second Part: Doc2vec 
NB: this code is likely to be improved !
'''
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import nltk
from nltk.corpus import stopwords
df=pd.read_csv("tripadvisor_museum_world.csv")
#df.head(10)
df1=df.copy()
top_rating=df1[['MuseumName','Rating']].sort_values('Rating',ascending=False)
#top_rating
rated=df1['Rating'].value_counts().to_frame().sort_values('Rating',ascending=False)
#Getting insight of museum ranking by exploring reviews score
plt.rcParams['figure.figsize']=(15,8)
rated.plot(kind='pie',label='pie chart',subplots=True,fontsize=14)
most_reviewed=df1[['MuseumName','ReviewCount']].sort_values(by='ReviewCount',ascending=False)
df.head(20).hist(color='Yellow')
#map plotting 
df1[['Langtitude','Latitude']].describe()
BB_zoom = (-157.958325, -45.876974, 67.273312, 176.260300)
nyc_map_zoom = plt.imread('http://upload.wikimedia.org/wikipedia/commons/8/81/BlankMap-World-FIFA.png') # map image 
lang=df1.loc[:,'Langtitude']
lat=df1.loc[:,'Latitude']
s=20
alpha=0.2
BB = BB_zoom
fig = plt.subplots(1, 1, figsize=(18,18))
plt.scatter(lang, lat, zorder=1, alpha=alpha, c='r', s=s)
plt.xlim((BB[0], BB[1]))
plt.ylim((BB[2], BB[3]))
plt.title('Museum in the world')
plt.imshow(nyc_map_zoom, zorder=0, extent=BB)

# e.g of the list:
'''
countries = [
{'timezones': ['Europe/Andorra'], 'code': 'AD', 'continent': 'Europe', 'name': 'Andorra', 'capital': 'Andorra la Vella'},
{'timezones': ['Asia/Kabul'], 'code': 'AF', 'continent': 'Asia', 'name': 'Afghanistan', 'capital': 'Kabul'}]

here the link: https://gist.github.com/Desperado/3293395#file-countryinfo-py
'''
df1.Address=df1.Address.apply(lambda x:re.findall(r'\w+',x))
#create new DF to allocate capital 
df3=pd.DataFrame(df1,columns=['Capital'])
for j in range(len(countries)):
    for index in range(0,len(df1)): 
        if countries[j]['capital'] in str(df1.Address[index]) :
            df3.loc[index,'Capital']=countries[j]['capital']     		
df4=pd.merge(df1,df3,left_index=True,right_index=True)
#Create function for cleansing Description Text
def review_to_wordlist(review, remove_stopwords=True):
    # Clean the text, with the option to remove stopwords.
    # Convert words to lower case and split them
    words = review.lower().split()
    # Optionally remove stop words (true by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    review_text = " ".join(words)
    # Clean the text
    review_text = re.sub(r"[^A-Za-z0-9(),!.?\'\`]", " ", review_text)
    review_text = re.sub(r"\'s", " 's ", review_text)
    review_text = re.sub(r"\'ve", " 've ", review_text)
    review_text = re.sub(r"n\'t", " 't ", review_text)
    review_text = re.sub(r"\'re", " 're ", review_text)
    review_text = re.sub(r"\'d", " 'd ", review_text)
    review_text = re.sub(r"\'ll", " 'll ", review_text)
    review_text = re.sub(r"'s", "", review_text) 
    review_text = re.sub(r",", " ", review_text)
    review_text = re.sub(r"\.", " ", review_text)
    review_text = re.sub(r"!", " ", review_text)
    review_text = re.sub(r"\(", " ( ", review_text)
    review_text = re.sub(r"\)", " ) ", review_text)
    review_text = re.sub(r"\?", " ", review_text)
    review_text = re.sub(r"\s{2,}", " ", review_text)
    review_text = re.sub(r"\d", " ", review_text) ### elminate all number (\d]
    words = review_text.split()
    # Shorten words to their stems
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in words]
    review_text = " ".join(stemmed_words)
    # Return a list of words
    return(review_text)	
# apply the function review_to_wordlist
df4['Description']=df4['Description'].apply(lambda x:review_to_wordlist(x,remove_stopwords=True))
#rename the columns Unnamed : 0
df4.rename(columns={'Unnamed: 0':'Id'},inplace=True)
# TfidfVectorizer and cosine_similarity
cv=TfidfVectorizer(max_features=None,
                    stop_words='english',
                    ngram_range=(1,1),
                    analyzer='word')
def cosine_sim(text1,text2):
    tfidf = cv.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]
def get_max(text2):
    Tfidf_score_df=[]
    S_score=pd.DataFrame(df4,columns=['Cosine_score'])
    for d in df4['Description']:
        Tfidf_score_line=cosine_sim(d,text2)
        Tfidf_score_df.append(Tfidf_score_line)
    S_score['Cosine_score']=Tfidf_score_df
    return S_score['Cosine_score']
'''
Unitary test:
get_max('exploration museum')// DF of cosine_similarity score
'''
# Get_recommendation function 
def get_recommendation (df4,text2):
    text2=review_to_wordlist(text2, remove_stopwords=True)
	MM=pd.DataFrame(df4,columns=['scores'])
    MM['scores']=get_max(text2)
    max_cosine=MM.max()
    df5=pd.merge(df4,MM, left_index=True, right_index=True)
    simus_scores = list(enumerate(df5[df5['scores']==max_cosine].index))
    return [df5.loc[simus_scores,'MuseumName']]
'''
Unitary test:
max_cosine=get_max('Egyptian mummies').max() // 
>>>0.08332471296185161
get_recommendation (df4,'Egyptian mummies')//
otherwise;
df5=pd.DataFrame(df4,columns=['scoress']) 
df5['scoress']=get_max('Egyptian mummies') 
df6=pd.merge(df4,df5, left_index=True, right_index=True)
print(df6[df6['scoress']==max_cosine].index) 
>>> Int64Index([1, 61, 151, 360, 715], dtype='int64')
df6['Description'].iloc[simus_scores]
>>> [1      British Museum
 61     British Museum
 151    British Museum
 360    British Museum
 715    British Museum
 Name: MuseumName, dtype: object]
'''

##### Second Attempt with Doc2vec and cosine_similarity

labeled_description=[]
for idx, row in df4.iterrows():
    labeled_lines=TaggedDocument(row['Description'].split(), df4[df4.index==idx].Id)
    labeled_description.append(labeled_lines)
#labeled_description
# function to get labeled senteces // I used TaggedDocument instead of LabeledSentenced 
def get_labeled_sentenced(text2):
    text2=review_to_wordlist(text2, remove_stopwords=True)
    return TaggedDocument(text2.split(),1)
#Building the model
model = Doc2Vec(dm = 1, min_count=1, window=10, vector_size=100, sample=1e-4, negative=10)
model.build_vocab(labeled_description)
#corpus_count 
print(model.corpus_count)
total_epoch = 20    
model.iter = total_epoch  
# start training
for epoch in range(total_epoch):
    model.train(labeled_description, total_examples=model.corpus_count,epochs=model.iter)
    print("Epoch #{} is complete.".format(epoch+1))
#Test Model
model.most_similar('modern')
'''
[('millennia', 0.9086717367172241),
 ('egypt', 0.88746178150177),
 ('let', 0.8620836734771729),
 ('enchant', 0.861074686050415),
 ('uniqu', 0.8526980876922607),
 ('foremost', 0.8518943786621094),
 ('drer', 0.8454443216323853),
 ('five', 0.8328124284744263),
 ('era', 0.8323265910148621),
 ('span', 0.8261540532112122)]
 '''
# calcul cosine_similarity_score
scores = []
for d in df4['Description']:
    score=model.n_similarity(d.split(),text2.split())
    scores.append(score)
doc2vec_scores=pd.DataFrame(scores, columns=['cosine_sim'])
doc2vec_scores=scores
df6=pf.merge(df4,doc2vec_scores, left_index=True, right_index=True)
score_max=df6['cosine_sim'].max()
# get the recommended museum
print(df6[df6['cosine_sim']==score_max].index)

''' Unitary test:
text2='mummies egypt' // Description written by the customer 
text2=get_labeled_sentenced(text2)
score_max=0.42438352 // max of the cosine_similarity (all row of df6, text2)
print('recommended museum:', [df6.loc[score_max,'MuseumName']])
>>>'British Museum' 

Next Steps:
    1-> exploring Fasttext/Glove and working on corpus augmentation for doc2vec.
	2->add all features related to the recommended museum (Address, phone number,...)
'''
	




