#!/usr/bin/env python
# coding: utf-8

# In[1]:


##Importing the packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


##Importing the dataset

comments = pd.read_csv('A:\DataScience Projects\YoutubeAnalysis\GBcomments.csv',error_bad_lines = False)


# In[3]:


comments.head()


# In[4]:


from textblob import TextBlob


# In[5]:


comments.isna().sum()


# In[6]:


comments.dropna(inplace = True)


# ### Sentiment Analysis

# In[7]:


#Checking the sentiment of each comment
pol = []

for i in comments['comment_text']:
    pol.append(TextBlob(i).sentiment.polarity)


# In[8]:


comments['polarity'] = pol


# In[9]:


comments.head(10)


# Positive polarity indicates a positive comment.
# Negative polarity indicates a negative comment.

# ### EDA for positive comments

# In[10]:


#Filtering the positve comments.
positive_comments = comments[comments['polarity']==1]


# In[11]:


positive_comments.head()


# In[12]:


from wordcloud import WordCloud,STOPWORDS


# In[13]:


stop = set(STOPWORDS)


# In[14]:


total_positive = "".join(positive_comments['comment_text'])


# In[15]:


total_positive


# In[16]:


pwc = WordCloud(height = 1400, width = 1000,stopwords = stop).generate(total_positive)


# In[17]:


plt.figure(figsize = (12,8))
plt.imshow(pwc)
plt.axis('off')


# ### EDA for negative comments

# In[18]:


##Filtering the negative comments.
negative_comments = comments[comments['polarity'] == -1]


# In[19]:


negative_comments.head()


# In[20]:


total_negative = "".join(negative_comments['comment_text'])


# In[21]:


total_negative


# In[22]:


nwc = WordCloud(height = 1400, width = 1000, stopwords = stop).generate(total_negative)


# In[23]:


plt.figure(figsize = (12,8))
plt.imshow(nwc)
plt.axis('off')


# ### What are the trending tags on youtube?

# In[24]:


videos = pd.read_csv('A:\DataScience Projects\YoutubeAnalysis/USvideos.csv',error_bad_lines = False)


# In[25]:


videos.head()


# In[26]:


all_tags = ' '.join(videos['tags'])


# In[27]:


all_tags


# In[28]:


import re


# In[29]:


##Removing special characters
f_tags = re.sub('[^a-zA-Z]'," ",all_tags)


# In[30]:


f_tags


# In[31]:


f_tags = re.sub(' +',' ',f_tags)


# In[32]:


f_tags


# In[33]:


wc = WordCloud(height = 1500,width = 1000,stopwords = set(STOPWORDS)).generate(f_tags)


# In[34]:


plt.figure(figsize = (12,8))
plt.axis('off')
plt.imshow(wc)


# ### Are Likes,Dislikes and Views correlated to each other?

# In[35]:


##Using a regression plot for 'Likes' and 'Views'
sns.regplot(x = 'views', y = 'likes',data = videos)
plt.title('Regression plot for Views & Likes')


# Likes increase with views which implies the number of likes have very strong relation with views.

# In[36]:


##Using a regression plot for 'Likes' and 'Views'
sns.regplot(x = 'views', y = 'dislikes',data = videos)
plt.title('Regression plot for Views & Dislikes')


# Dislikes not necessarily increase when the views are increasing.

# #### Correlation Matrix

# In[37]:


df_corr = videos[['views','likes','dislikes']]


# In[38]:


df_corr.corr()


# In[39]:


sns.heatmap(df_corr.corr(),annot=True)


# ### Analysis of the emojis in the comment section

# In[40]:


import emoji


# In[41]:


len(comments)


# In[42]:


str=''
for i in comments['comment_text']:
    list=[c for c in i if c in emoji.UNICODE_EMOJI['en']]
    for ele in list:
        str=str+ele


# In[43]:


len(str)


# In[44]:


print(str)


# In[45]:


uni_emoji = {}
for i in set(str):
    uni_emoji[i] = str.count(i)


# In[46]:


uni_emoji


# In[47]:


##Sorting the dictionary.

final = {}
for key,value in sorted(uni_emoji.items(), key = lambda item:item[1]):
    final[key] = value


# In[48]:


final 


# In[49]:


keys = [*final.keys()]


# In[50]:


values = [*final.values()]


# In[51]:


keys


# In[52]:


values


# In[53]:


##Top 30 emojis
df=pd.DataFrame({'chars':keys[-30:],'num':values[-30:]})


# In[54]:


df


# In[55]:


import plotly.graph_objs as go
from plotly.offline import iplot


# In[56]:


trace=go.Bar(x=df['chars'],y=df['num'])

iplot([trace])


# We can clearly see that laughing emojis is used the most in the comments section.

# In[ ]:




