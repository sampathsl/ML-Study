#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("Employee_Turn_Over3.csv")
df.shape


# In[36]:


df.isna().sum()


# In[37]:


df['Is Terminated'].value_counts()


# In[38]:


df['Is Terminated'].value_counts(normalize=True) * 100


# In[18]:


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


# In[39]:


# plt.subplots(figsize=(20,8))
# ax = sns.countplot(x='2nd Month OT', hue='Is Terminated', data= df, palette = 'colorblind')
# ax.set_xlabel('Month OT')
# ax.set_ylabel('Number of Employee')
# ax.set_title("Distribution of Employee Termination Over Salary Category")

df.plot.hist(y='2nd_Month_OT', bins=1000);


# In[10]:


sns.catplot(x="Age_Joined",y="CivilStatus_cat",kind='box',data=df)


# In[41]:


# sns.catplot(x="Age_Joined",kind='box',data=df)

#mtcars.plot(kind='scatter',x='mpg',y='drat')
#plt.subplot(212)
#df.2nd_Month_Gross_Salary.plot(kind='density')

plt.subplots(figsize=(20,8))
sns.stripplot(x="Salary_Category_3",y="2nd_Month_Gross_Salary",data=df, jitter=True,hue='Is Terminated',palette='Set1')


# In[46]:


sns.catplot(x="Salary_Category_3",kind='box',data=df)


# In[44]:


# Plot
plt.figure(figsize=(40,40), dpi= 80)
sns.heatmap(df.corr(), xticklabels=df.corr().columns, yticklabels=df.corr().columns, cmap='RdYlGn', center=0, annot=True)

# Decorations
plt.title('Correlogram of mtcars', fontsize=22)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# In[ ]:




