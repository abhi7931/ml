#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# print(os.listdir('../input/bank-customer-churn-modeling'))
df= pd.read_csv('Churn_Modelling.csv')


# In[4]:


df.shape


# In[5]:


df.sample(5)


# In[6]:


df.info()


# In[7]:


df.drop(['RowNumber','CustomerId','Surname'],axis=1,inplace=True)


# In[8]:


df.dtypes


# In[9]:


df.Exited.value_counts()


# In[10]:


df.isna().sum()


# In[11]:


cat_cols=['Geography','Gender']
num_cols=[col for col in df.columns if col not in cat_cols]


# In[12]:


for col in cat_cols:
    print(f'{col} : {df[col].unique()}')


# In[13]:


df['Gender'].replace({'Female':1,'Male':0},inplace=True)


# In[14]:


df=pd.get_dummies(data=df, columns=['Geography'])


# In[15]:


tenure_exited_0=df[df.Exited==0].Tenure
tenure_exited_1=df[df.Exited==1].Tenure

plt.figure(figsize=(10,8))
plt.xlabel('T enure')
plt.ylabel('Number of Customers Exited')
plt.title('Bank Customer Churn prediction visualization')
plt.hist([tenure_exited_1,tenure_exited_0], color=['green','red'], label=['Exited-yes','Exited-No'])
plt.legend()


# In[16]:


creditscore_exited_0=df[df.Exited==0].CreditScore
creditscore_exited_1=df[df.Exited==1].CreditScore

plt.figure(figsize=(10,8))
plt.xlabel('Credit Score')
plt.ylabel('Number of Customers Exited')
plt.title('Bank Customer Churn prediction visualization')
plt.hist([creditscore_exited_1,creditscore_exited_0], color=['green','red'], label=['Exited-yes','Exited-No'])
plt.legend()


# In[17]:


df.info()


# In[18]:


# Scaling
cols_to_scale=['CreditScore','Tenure','Balance','NumOfProducts','EstimatedSalary','Age']

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

df[cols_to_scale]=scaler.fit_transform(df[cols_to_scale])


# In[19]:


# Training
x=df.drop('Exited',axis=1)
y=df.Exited

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=15,stratify=y)


# In[20]:


def ANN(xtrain,xtest,ytrain,ytest,loss,weight):
    model=keras.Sequential([
    keras.layers.Dense(20,input_shape=(12,),activation='relu'),
    keras.layers.Dense(1,activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                 loss=loss,
                 metrics=['accuracy'])
    
    if weight==-1:
        model.fit(xtrain,ytrain,epochs=100)
    else:
        model.fit(xtrain,ytrain,epochs=100,class_weight=weight)
    print()
    print(model.evaluate(xtest,ytest))
    print()
    ypred= model.predict(xtest)
    ypred=np.round(ypred)
    print()
    print(classification_report(ytest,ypred))
        
    return ypred


# In[21]:


ypred=ANN(xtrain,xtest,ytrain,ytest,'binary_crossentropy',-1)


# In[22]:


cm=tf.math.confusion_matrix(labels=ytest,predictions=ypred)
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True,fmt='d')
plt.xlabel('predicted')
plt.ylabel('Truth')


# In[ ]:




