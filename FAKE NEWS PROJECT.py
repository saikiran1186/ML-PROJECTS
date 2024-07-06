#!/usr/bin/env python
# coding: utf-8

# In[110]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as split


# In[111]:


df=pd.read_csv(r"C:\Users\saikiran\Downloads\fakenews (1).csv")
df


# In[112]:


df1=df.copy()
df1


# In[113]:


df1.shape


# ### EXPLORATORY DATA ANALYSIS

# In[114]:


df.info()


# In[115]:


## checking for the null values
df1.isnull().sum()


# In[116]:


dup_rows=df1[df1.duplicated()]
dup_rows


# In[117]:


df1=df1.drop_duplicates().reset_index(drop=True)
df1


# In[118]:


df1["label"].value_counts(normalize=True)


# In[119]:


sns.countplot(data=df1,x="label")


# In[120]:


import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


# In[121]:


def preprocess(text):
    #Removing special characters and digits
    sentence=re.sub("[^a-zA-Z]"," ",text)
    
    #converting into lower case
    sentence=sentence.lower()
    
    return sentence


# In[122]:


df1["processed_text"]=df1["text"].apply(lambda x:preprocess(x))
df1


# In[126]:


df1 = df1.drop_duplicates(subset='processed_text')
df1.reset_index(drop=True,inplace=True)


# In[127]:


df1


# In[144]:


import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


# In[146]:


lemmatizer=WordNetLemmatizer()


# In[145]:


nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('punkt')


# In[134]:


#identifying the target and input variables
y=df1["label"]
x=df1[["processed_text"]]


# In[135]:


# Splitting the text into train test split
x_train,x_test,y_train,y_test=split(x,y,train_size=0.75,random_state=42)


# In[136]:


print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


# In[149]:


def cleaning(processed_text,flag):
    #removing the stop words
    tokens=processed_text.split()
    
    clean_tokens=[i for i in tokens if not i in stopwords.words("english")]
    
    #lemmatizing
    if flag=="lemma":
        clean_tokens=[lemmatizer.lemmatize(word) for word in clean_tokens]
    
    return pd.Series([" ".join(clean_tokens),len(clean_tokens)])


# In[ ]:





# In[153]:


temp_df=x_train["processed_text"].apply(lambda x:cleaning(x,"lemma"))
temp_df.head()


# In[155]:


temp_df.columns=["clean_text_lemma","text_length_lemma"]
temp_df.head()


# In[156]:


x_train=pd.concat([x_train,temp_df],axis=1)
x_train.head()


# ### creating the wordcloud(visualization)

# In[157]:


get_ipython().system('pip install wordcloud')


# In[158]:


from wordcloud import WordCloud


# In[159]:


y_train


# In[166]:


fake_df=x_train.loc[y_train==1,:]
fake_df


# In[167]:


words=" ".join(fake_df["clean_text_lemma"])
print(words[:50])


# In[168]:


fake_df=x_train.loc[y_train==1,:]

words=" ".join(fake_df["clean_text_lemma"])


# In[169]:


fake_wordcloud=WordCloud(stopwords=stopwords.words("english"),
                        background_color="black",
                        width=1600,
                        height=800).generate(words)


# In[170]:


plt.figure(1,figsize=(30,20))
plt.imshow(fake_wordcloud)
plt.axis("off")
plt.show()

-- Observation:in the generated WordCloud terms such as "said","one","time" dominate visually
                This says that the discourse around fake news in our dataset frequently 
# In[172]:


fake_df=x_train.loc[y_train==0,:]

words=" ".join(fake_df["clean_text_lemma"])


# In[173]:


fake_wordcloud=WordCloud(stopwords=stopwords.words("english"),
                        background_color="black",
                        width=1600,
                        height=800).generate(words)


# In[174]:


plt.figure(1,figsize=(30,20))
plt.imshow(fake_wordcloud)
plt.axis("off")
plt.show()

-- Observation:in the generated WordCloud terms such as "said","one","show","time" dominate visually
                This says that the discourse around real news in our dataset frequently 
# ### Converting text to numerical vectors using TF-IDF

# In[176]:


from sklearn.feature_extraction.text import TfidfVectorizer

vocab=TfidfVectorizer()

x_train_tf_idf=vocab.fit_transform(x_train["clean_text_lemma"])


# In[177]:


x_train_tf_idf


# In[178]:


print("Total unique words",len(vocab.vocabulary_))
print("Type of train features",type(x_train_tf_idf))
print("shape of the input data",x_train_tf_idf.shape)


# In[179]:


print(x_train_tf_idf.toarray())


# In[180]:


from sys import getsizeof

print(type(x_train_tf_idf))
print(getsizeof(x_train_tf_idf.toarray()),"Bytes")


# ## preprocessing test data

# In[181]:


x_test.head()


# In[183]:


temp_df=x_test["processed_text"].apply(lambda x:cleaning(x,"lemma"))
temp_df


# In[184]:


temp_df.columns=["clean_text_lemma","text_length_lemma"]
temp_df.head()


# In[185]:


x_test=pd.concat([x_test,temp_df],axis=1)
x_test.head()


# In[193]:


x_test_tf_idf=vocab.transform(x_test["clean_text_lemma"])
x_test_tf_idf


# In[190]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[195]:


#Logistic regression
LR_classifier = LogisticRegression()
LR_classifier.fit(x_train_tf_idf,y_train)


# In[200]:


y_test_pred_LR=LR_classifier.predict(x_test_tf_idf)


# In[202]:


y_test_pred_LR


# In[199]:


from sklearn import metrics


# In[203]:


metrics.accuracy_score(y_test,y_test_pred_LR)


# In[204]:


#Decision tree
DT_classifier = DecisionTreeClassifier()

DT_classifier.fit(x_train_tf_idf,y_train)


# In[206]:


y_test_pred_DT=DT_classifier.predict(x_test_tf_idf)
y_test_pred_DT


# In[208]:


metrics.accuracy_score(y_test,y_test_pred_DT)


# In[209]:


#svc
svm_classifier = SVC()

svm_classifier.fit(x_train_tf_idf,y_train)


# In[210]:


y_test_pred_svm=svm_classifier.predict(x_test_tf_idf)
y_test_pred_svm


# In[211]:


metrics.accuracy_score(y_test,y_test_pred_svm)


# In[212]:


#RANDOMFOREST CLASSIFIER
RF_classifier = RandomForestClassifier()

RF_classifier.fit(x_train_tf_idf,y_train)


# In[213]:


y_test_pred_RF=RF_classifier.predict(x_test_tf_idf)
y_test_pred_RF


# In[214]:


metrics.accuracy_score(y_test,y_test_pred_RF)


# In[217]:


#KNN CLASSIFIER
KNN_classifier = KNeighborsClassifier()

KNN_classifier.fit(x_train_tf_idf,y_train)


# In[218]:


y_test_pred_KNN=KNN_classifier.predict(x_test_tf_idf)
y_test_pred_KNN


# In[219]:


metrics.accuracy_score(y_test,y_test_pred_KNN)


# In[220]:


d={
    "Algo":["Logistic Regression","DecisionTreeClassifier"," SVC","RandomForestClassifier","KNeighborsClassifier"],
    "Accuracy_score":[0.775993237531699,0.6804733727810651,0.7776838546069316,0.7658495350803043,0.705832628909552]
}
pd.DataFrame(d)


# ## insights
-- The SVC (Support Vector Classifier) shows the highest accuracy score at 0.777684. Therefore, based on the metric of accuracy    alone, the SVC is the best-fitted model among those listed.
# In[ ]:




