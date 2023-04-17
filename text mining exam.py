#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from nltk.corpus import stopwords
import re


# In[3]:


df = pd.read_csv('trumptweets.csv')


# In[4]:


df.head()


# In[ ]:





# In[5]:


##chain is the string and contentis the speach

speaches = df.content


# In[6]:


#important 100%
speaches_string = ''' '''.join(i for i in speaches)
speaches_string


# In[7]:


test = 'Hello World! [Applause] How are you ?'
text = test


# In[8]:


def remove_brackets(string):
    new_string = []
    import re
    new_string = re.split('[\?\[\]\!_ ]+', string)
    return new_string


# In[9]:


remove_brackets(text)


# In[53]:


from PIL import Image
import numpy as np
mask = np.array (Image.open ("heart.png"))


# In[11]:


remove_brackets(speaches_string)


# In[12]:


from nltk.corpus import stopwords
swf1 = set(stopwords.words('english'))


# In[13]:


#swf2 = ["?", "i've", '@']


# In[14]:


swf1.update(["?", "i've", '@'])


# In[15]:


from nltk.tokenize import word_tokenize


# In[16]:


from nltk.tokenize import TweetTokenizer


# In[17]:


tokenizer = TweetTokenizer()


# In[18]:


tokens = tokenizer.tokenize(speaches_string)


# In[19]:


tokens


# In[20]:


#str_tkns = ''' '''.join(i for i in tokens)
#str_tkns


# In[21]:


#new_tkns = remove_brackets(str_tkns)
#new_tkns


# In[22]:


from nltk.probability import FreqDist
fdist = FreqDist(tokens)
fdist


# In[ ]:


#total of all tokens is 1017962 
fdist.N()


# In[23]:


import nltk
text2 = nltk.Text(tokens)
len(text2)


# In[25]:


# Add stopwords 
swf1.update(["?", "i've", '@','.',',','"',':','#','-','’','&'])

# define the function stop_words_filtering 
def stop_words_filetring(words) : 
    tokenss = []
    for i in words:
        if i not in swf1:
            tokenss.append(i)
    return tokenss
new_of_new = stop_words_filetring(tokens)
new_of_new


# In[26]:


fdist = FreqDist(new_of_new)
fdist


# In[27]:


fdist.N()


# In[31]:


get_ipython().system('pip install wordcloud')


# In[32]:


# WordCloud phase: tokens new_of_new

from wordcloud import WordCloud


# In[36]:


wc = WordCloud(background_color="white", max_words=100, stopwords=swf1, max_font_size=90, random_state=42, collocations = False)


# In[71]:


str_tokens = ""
for i in new_of_new : 
    str_tokens += i


# In[72]:


import matplotlib.pyplot as plt
plt.figure(figsize = (10,6))
wc.generate(str_tokens)
plt.imshow(wc)
plt.show()


# In[50]:


from wordcloud import ImageColorGenerator 


# In[54]:


img_color = ImageColorGenerator(mask)


# In[57]:


color_func=img_color
color_func.recolor()
plt.imshow(wc,interpolation= 'bilinear')
plt.axis('off')
plt.show


# In[58]:


from sklearn.model_selection import train_test_split
X = df.content
y = df.Sentiment
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[67]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


# In[61]:


vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train).todense()
X_test = vectorizer.transform(X_test).todense()


# In[64]:


clf = GradientBoostingClassifier(n_estimators = 100,learning_rate = 1.0,max_depth = 1, random_state = 0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred


# In[68]:



# Calculation and display of classification_report
print( classification_report(y_test, y_pred) )

# Calculation and display of the confusion matrix
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
confusion_matrix


# In[69]:


dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
y_pred2 = dt_clf.predict(X_test)
y_pred2


# In[70]:


print( classification_report(y_test, y_pred2) )

# Calculation and display of the confusion matrix
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
confusion_matrix


# In[ ]:





# In[ ]:




