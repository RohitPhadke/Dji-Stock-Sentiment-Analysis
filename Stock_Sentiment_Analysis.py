#!/usr/bin/env python
# coding: utf-8

# ## Stock Sentiment Analysis using News Headlines

# In[175]:


#Perform EDA
import numpy as np
import pandas as pd
# For visualizations
import matplotlib.pyplot as plt
# For regular expressions
import re
#Machine Learning Algorithms
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_auc_score
from sklearn import metrics
#Library for deploying the model
import pickle


# In[176]:


#Reading the data
df=pd.read_csv('Data.csv', encoding = "ISO-8859-1")


# In[177]:


#Check the size of data
print("Shape of data=>",df.shape)


# In[178]:


df.head()


# In[179]:


#Check null values
df.isnull().sum()


# In[180]:


#Drop null values
df.dropna(inplace=True)
df.isnull().sum()


# In[181]:


df.info()


# In[182]:


#Check how many are positive and how many are negative news headlines
df['Label'].value_counts().plot(kind='bar');
#So this is a balanced dataset


# In[183]:


#Splitting data into train and test
train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']


# In[184]:


#Check the train test dataset
print("Training set has "+str(sum(train['Label'] == 1))+ " samples with label equal to 1")
print("Training set has "+str(sum(train['Label'] == 0))+ " samples with label equal to 0")
print("Testing set has "+str(sum(test['Label'] == 1))+ " samples with label equal to 1")
print("Testing set has "+str(sum(test['Label'] == 0))+ " samples with label equal to 0")


# In[185]:


#Data Preprocessing
# Removing punctuations
data=train.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)

# Renaming column names for ease of access
list1= [i for i in range(25)]
new_Index=[str(i) for i in list1]
data.columns= new_Index
data.head(5)


# In[186]:


# Convertng headlines to lower case
for index in new_Index:
    data[index]=data[index].str.lower()
data.head(1)


# In[187]:


' '.join(str(x) for x in data.iloc[1,0:25])


# In[188]:


headlines = []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))


# In[189]:


headlines[0]


# In[190]:


new_headlines = []
for i in headlines:
    j = i.replace('  ','')
    new_headlines.append(j)


# In[191]:


new_headlines[0]


# In[192]:


## implement BAG OF WORDS
countvector=CountVectorizer(ngram_range=(2,2))
traindataset=countvector.fit_transform(new_headlines)


# In[193]:


#Creating a pickle file for CountVectorizer
pickle.dump(countvector, open('stocksentimentcv.pkl', 'wb'))


# In[194]:


#Applying various classification models to determine the most accurate model
LG = LogisticRegression()
DT = DecisionTreeClassifier()
MNB = MultinomialNB()
SVM = svm.SVC()
RF = RandomForestClassifier()

models = [LG, DT, MNB, SVM, RF]
## Predict for the Test Dataset
test_transform= []
for row in range(0,len(test.index)):
  test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
  test_dataset = countvector.transform(test_transform)


# In[195]:


# Try various classification models
Model_accuracy = []
pred = []
for classifier in models:
 classifier.fit(traindataset,train['Label'])
 predictions = classifier.predict(test_dataset)
 pred.append(predictions)
 score=accuracy_score(test['Label'],predictions)
 Model_accuracy.append(score)


# In[196]:


#Create dictionary of Models and their corresponding accuracy
best_model = {} 
for key,value in zip(models,Model_accuracy): 
        best_model[key] = value
print(best_model)


# In[197]:


#Most accurate model
print(str(max(best_model.items(), key=operator.itemgetter(1))[0]) +" has highest accuracy of "+ str(max(best_model.values())))


# In[198]:


#Check model predictions
matrix=confusion_matrix(test['Label'],pred[0])
print(matrix)
score=accuracy_score(test['Label'],pred[0])
print(score)
report=classification_report(test['Label'],pred[0])
print(report)


# In[199]:


#Creating a pickle file for model deployment
LG = LogisticRegression()
model = LG.fit(traindataset,train['Label'])
filename = 'stocksentimentclassifier.pkl'
pickle.dump(model, open(filename, 'wb'))


# In[200]:


from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd 
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                min_font_size = 5).generate(new_headlines[0]) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 


# In[ ]:




