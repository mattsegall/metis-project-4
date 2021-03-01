#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import re
import sys
from tqdm import tqdm


# In[4]:


sys.path.append("/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages")


# In[5]:


from fastai.text import *


# In[6]:


#read in data
df = pd.read_csv('realdonaldtrump.csv')


# In[7]:


#im really only interested in trump tweets after he became a political figure, 
#so I'm dropping tweets from before he anncounced his canidacy
mask = (df['date'] > '2015-06-16')
df = df.loc[mask]


# In[8]:


data = df["content"]
data.columns = ["text"]
train = data

train.to_csv("./trump_tweet_gen_train.csv")


# In[9]:


def remove_punc(text):
    new_text = re.sub(r'[^\w\s]', '', text) 
    return new_text


# In[10]:


#remove punctuation
for i in train.index:
    train.loc[i] = remove_punc(train.loc[i])


# In[11]:


train


# In[12]:


train = pd.DataFrame(train)
train


# In[13]:


data = (TextList.from_df(train, cols='content')
                .split_by_rand_pct(0.1)
                .label_for_lm()  
                .databunch(bs=48))


# In[14]:


data.show_batch()


# In[15]:


#instantiate model
model = language_model_learner(data, AWD_LSTM, drop_mult=0.1, model_dir = '/Users/user/desktop/metis/projects/metis-project-4/tweet_generator_model')

#find learning rate
model.lr_find()

# Fit the model 
model.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))


# In[16]:


model.recorder.plot()


# In[17]:


print(model.predict("I think CNN is", 20, temperature=0.75))


# In[18]:


print(model.predict("I think Fox News is", 20, temperature=0.75))


# In[19]:


print(model.predict("Russia has", 20, temperature=0.75))


# In[20]:


print(model.predict("Mueller Report", 20, temperature=0.75))


# In[21]:


print(model.predict("Impeachment", 20, temperature=0.75))


# In[22]:


print(model.predict("COVID is a ", 20, temperature=0.75))


# In[ ]:





# In[24]:


trump_tweet_predict('North Korea', 10)


# In[25]:


import pickle


# In[31]:


# save the model to disk
model.export('trump_tweet_gen.pkl')


# In[32]:


def trump_tweet_predict(starter, n):
    x = load_learner('', 'trump_tweet_gen.pkl')
    result = x.predict(starter, n, temperature=0.75)
    return print(result)


# In[ ]:





# In[ ]:




