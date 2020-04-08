#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("Employee_Turn_Over3.csv")
df.shape


# In[33]:


df.isna().sum()


# In[34]:


df['Is Terminated'].value_counts()


# In[35]:


df['Is Terminated'].value_counts(normalize=True) * 100


# In[28]:


# sns.catplot('Salary_Category_3', data=df, aspect=3, kind='count', hue='Is Terminated', palette=['C1', 'C0']).set_ylabels('Number of Employees')
import matplotlib.pyplot as plt

total = df.shape[0]
plt.subplots(figsize=(20,8))
ax = sns.countplot(x='Salary_Category_3', hue='Is Terminated', data= df, palette = 'colorblind')
ax.set_xlabel('Salary Category')
ax.set_ylabel('Number of Employee')
ax.set_title("Distribution of Employee Termination Over Salary Category")
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total * 100),
            ha="center")


# In[31]:


# plt.subplots(figsize=(20,8))
# ax = sns.countplot(x='2nd Month OT', hue='Is Terminated', data= df, palette = 'colorblind')
# ax.set_xlabel('Month OT')
# ax.set_ylabel('Number of Employee')
# ax.set_title("Distribution of Employee Termination Over Salary Category")

df.plot.hist(y='2nd Month OT', bins=1000);


# In[ ]:




