#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 15:39:35 2021

@author: arifshakil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
import os
import shutil
import sys


# In[2]:


from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
#from mlxtend.plotting import plot_learning_curves
import datetime


# In[3]:


datetime.datetime.now()
datetime.datetime(2009, 1, 6, 15, 8, 24, 78915)
print("Start Time:", datetime.datetime.now())


# In[4]:


dataframes = []
for i in range(1, 37):
  path = 'Dataset/'+str(i)+'/' # 1/
  for file in os.listdir(path):
   # print(file)
    data = pd.read_csv(path+file,sep='\s+') #1_raw_data_12-10_26.04.16.txt
    dataframe = pd.DataFrame(data)
    dataframes.append(dataframe)
     #os.close(0)       
result1 = pd.concat(dataframes)
result1= pd.DataFrame(result1)
print("Data import complete:", datetime.datetime.now())


# In[5]:


result1.shape


# In[6]:


result1.head(5)


# In[7]:


result1['class'].value_counts()


# In[11]:




# In[12]:


#Working with missing value
result1.replace([np.inf, -np.inf], np.nan)
result1.dropna(inplace=True)
result1['class'] = result1['class'].astype(int)

result1.head(5)
result1['class'].value_counts()

# In[13]:


#Removing zero values
result = result1[result1['class'] != 0]
result.head(5)

# In[14]:


result.shape


# In[15]:


result['class'].value_counts()


# In[16]:

result.loc[result['class'] == 7, 'class'] = 0
#result['class'].replace({7:0}, 0)
result.head(5)
result['class'].value_counts()

# In[17]:


result['class'].value_counts()

result[result['class']==0].head(5)


# In[18]:


#Creating Matrix of Features
x = result.iloc[:, :-1 ].values
y = result['class'].values


# In[19]:


print("X = " ,x.shape)
print("Y = " ,y.shape)

idx = np.array(list(range(len(y)))) 
np.random.shuffle(idx)

x = x[idx]
y = y[idx]

# In[20]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 0)



# In[21]:


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
x_train= sc_x.fit_transform(x_train)
x_test= sc_x.fit_transform(x_test)

np.save('Data/x_train.npy', x_train)
np.save('Data/x_test.npy', x_test)
np.save('Data/y_train.npy', y_train)
np.save('Data/y_test.npy', y_test)


# In[40]:


print("Train Set count:",np.unique(y_train, return_counts=True))
print("Test Set count:", np.unique(y_test, return_counts=True))
np.savetxt("y_train_unique.txt",np.unique(y_train, return_counts=True))
np.savetxt("y_test_unique.txt",np.unique(y_test, return_counts=True))

# In[26]:


len(np.unique(y_train))