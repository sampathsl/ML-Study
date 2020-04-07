#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("Employee_Turn_Over3.csv")
df.shape


# In[19]:


df.isna().sum()


# In[20]:


df['Is Terminated'].value_counts()


# In[21]:


df['Is Terminated'].value_counts(normalize=True) * 100


# In[22]:


sns.catplot('Salary_Category_3', data=df, aspect=3, kind='count', hue='Is Terminated', palette=['C1', 'C0']).set_ylabels('Number of Employees')


# In[ ]:




