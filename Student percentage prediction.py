#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data=pd.read_csv(r'C:\Users\OWNER\OneDrive\Desktop\percentage.csv')


# In[6]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[26]:


X=data['Hours']
Y=data['Score']
arr1 =X.values
arr2 =Y.values
X = arr1.reshape((18, 1))
Y = arr2.reshape((18, 1))


# In[29]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.10,shuffle=True,random_state=50)


# In[33]:


LR = LinearRegression()
# fitting the training data
LR.fit(X_train,y_train)
y_prediction =  LR.predict(X_test)


# In[40]:


from sklearn.metrics import r2_score
score=r2_score(y_test,y_prediction)
print('linear regression socre is ',score)



# In[44]:


#predicting the score 
dataset = np.array(9.25)
dataset = dataset.reshape(-1, 1)
pred = LR.predict(dataset)
print("If the student studies for 9.25 hours/day, the score is {}.".format(pred))


# In[ ]:




