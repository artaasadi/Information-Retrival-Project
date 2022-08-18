#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install hazm')
get_ipython().system('pip install parsivar')


# In[114]:


import hazm as hazm
from parsivar import FindStems
import re
import pandas as pd
from tqdm import tqdm
import json
import math
import numpy as np


# ### Data Preprocess

# In[2]:


persian = open('persian-stopwords-master/persian', 'r', encoding='utf8')
persian = persian.read().split('\n')
verbal = open('persian-stopwords-master/verbal', 'r', encoding='utf8')
verbal = verbal.read().split('\n')
nonverbal = open('persian-stopwords-master/nonverbal', 'r', encoding='utf8')
nonverbal = nonverbal.read().split('\n')
chars = open('persian-stopwords-master/chars', 'r', encoding='utf8')
chars = chars.read().split('\n')
short = open('persian-stopwords-master/short', 'r', encoding='utf8')
short = short.read().split('\n')
stop_words = persian + verbal + nonverbal + chars + short


# In[3]:


def preprocess(text):
    text = re.sub(r'http\S+', '', text)
    normalizer = hazm.Normalizer()
    text = normalizer.normalize(text)
    my_tokenizer = hazm.WordTokenizer()
    text = my_tokenizer.tokenize(text)
    text  = [word for word in text if word.lower() not in stop_words]
    my_stemmer = FindStems()
    return [my_stemmer.convert_to_stem(word) for word in text]


# In[4]:


try:
    data = pd.read_json('preprocessed_data.json')
except:
    data = pd.read_json('IR_data_news_12k.json').T
    for i in tqdm(range(len(data))) :
        data.iloc[i].content = preprocess(data.iloc[i].content)

    data.to_json('preprocessed_data.json')


# ### Inverted Index

# In[7]:


try:
    json_file = open('inverted_list.json')
    inverted_list = json.load(json_file)
except:
    inverted_list = {}
    for i in tqdm(range(len(data))) :
        content = data.iloc[i].content
        for j in range(len(content)):
            if content[j] in inverted_list:
                if str(i) in inverted_list[content[j]]:
                    inverted_list[content[j]][str(i)].append(j)
                else:
                    inverted_list[content[j]][str(i)] = [j]
            else:
                inverted_list[content[j]] = {
                    str(i): [j]
                }
    json_file = open('inverted_list.json', 'w')
    json.dump(inverted_list, json_file)


# ### rates

# In[28]:


for i in inverted_list:
    for j in inverted_list[i]:
        inverted_list[i][j]= inverted_list[i][j], ((1 + math.log(len(inverted_list[i][j]))) * math.log(len(data)/len(inverted_list[i])))


# ### save norms

# In[162]:


norms = {}
for i in tqdm(range(len(data))):
    vector = []
    for word in inverted_list:
        if str(i) in inverted_list[word]:
            vector.append(inverted_list[word][str(i)][1])
        else:
            vector.append(0)
    norms[str(i)] = np.linalg.norm(vector)


# ### Get Query

# In[163]:


def cosign_similarity(a, b, norm):
    return np.dot(a,b)/(np.linalg.norm(a)*norm)


# In[164]:


def query(query):
    query = preprocess(query)
    query_weights = []
    unique= list(set(query))
    for word in unique:
        query_weights.append((1 + math.log(query.count(word))) * math.log(len(data)/len(inverted_list[word])))
    docs= {}
    words_counter= 0
    for word in unique:
        if word in inverted_list:
            for i in inverted_list[word]:
                if i not in docs:
                    docs[i]= [*[0]*words_counter, inverted_list[word][i][1]]
                else:
                    docs[i]= [*docs[i], inverted_list[word][i][1]]
        words_counter+= 1
        for i in docs:
            if len(docs[i]) < words_counter:
                docs[i]= [*docs[i], 0]
    for i in docs:
        docs[i]= cosign_similarity(query_weights, docs[i], norms[i])
    docs = sorted(docs.items(), key= lambda x: x[1], reverse=True)
    return docs


# In[256]:


result= query("سانتریفیوژ IR-6 در فردو")
for (i, rate) in result[:10]:
    print(data.iloc[int(i)])


# ### now lets make a championList

# In[237]:


champs= inverted_list.copy()
k = 300
for i in champs:
    champs[i] = dict(sorted(champs[i].items(), key= lambda x: x[1][1], reverse=True)[:k])


# In[238]:


def query2(query):
    query = preprocess(query)
    query_weights = []
    unique= list(set(query))
    for word in unique:
        query_weights.append((1 + math.log(query.count(word))) * math.log(len(data)/len(champs[word])))
    docs= {}
    words_counter= 0
    for word in unique:
        if word in champs:
            for i in champs[word]:
                if i not in docs:
                    docs[i]= [*[0]*words_counter, champs[word][i][1]]
                else:
                    docs[i]= [*docs[i], champs[word][i][1]]
        words_counter+= 1
        for i in docs:
            if len(docs[i]) < words_counter:
                docs[i]= [*docs[i], 0]
    for i in docs:
        docs[i]= cosign_similarity(query_weights, docs[i], norms[i])
    docs = sorted(docs.items(), key= lambda x: x[1], reverse=True)
    return docs


# In[257]:


result = query2("تحریم هسته ای ایران")
for (i, rate) in result[:10]:
    print(data.iloc[int(i)])


# In[ ]:
