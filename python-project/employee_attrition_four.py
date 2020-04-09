#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Employee_Turn_Over4.csv")
df.shape


# In[10]:


df.isna().sum()


# In[12]:


plt.figure(figsize=(10,10))
corr = df.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,
            cmap=sns.diverging_palette(220, 10, as_cmap=True), annot=True)


# In[ ]:
