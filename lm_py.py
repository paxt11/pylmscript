#!/usr/bin/env python
# coding: utf-8

# # Linear Modeling in Python

# In[11]:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


# In[12]:


df = pd.read_csv("regrex1.csv")


# In[13]:


df.head()


# In[14]:


plt.scatter(df['x'],df['y'])
plt.xlabel('dfx')
plt.ylabel('dfy')
plt.savefig('py_orig.png')

# In[15]:


model = LinearRegression()


# In[16]:


x = np.array(df['x']).reshape(-1,1)
y = np.array(df['y'])


# In[17]:


model.fit(x,y)


# In[18]:


r_sq = model.score(x,y)
print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}")


# In[19]:


y_pred = model.predict(x)


# In[20]:


plt.scatter(x,y)
plt.plot(x,y_pred)
plt.xlabel('dfx')
plt.ylabel('dfy')
plt.savefig('py_lm.png')

# In[ ]:




