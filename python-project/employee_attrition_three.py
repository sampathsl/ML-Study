#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[226]:


df = pd.read_csv("New_Datset_Employee_Turnover_Old.csv")
# df = pd.read_csv("New_Datset_Employee_Turnover_Original.csv")
# df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
# print(df)
# df.head()


# In[227]:


df.shape


# In[228]:


df.dtypes


# In[229]:


df.isna().sum()


# In[230]:


df.isnull().values.any()


# In[231]:


df['Is Terminated'].value_counts()


# In[232]:


df.describe()


# In[233]:


import matplotlib.pyplot as plt
plt.title('Distribution of Employee Termination')
plt.xlabel('Number of Axles')
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.countplot(df['Is Terminated'])


# In[134]:


(1087 - 539) / 1087


# In[223]:


#Show the number of employees that left and stayed by age

# plt.subplots(figsize=(25,12))
# sns.countplot(x='Age Joined', hue='Is Terminated', data= df, palette = 'colorblind')
sns.catplot('Age Joined', data=df, aspect=3, kind='count', hue='Is Terminated', palette=['C1', 'C0']).set_ylabels('Number of Employees')


# In[167]:


#Show the number of employees that left and stayed by age
import matplotlib.pyplot as plt
#plt.subplots(figsize=(20,8))
#sns.countplot(x='Age Category', hue='Is Terminated', data= df, palette = 'colorblind')
sns.catplot('Age Category', data=df, aspect=3, kind='count', hue='Is Terminated', palette=['C1', 'C0']).set_ylabels('Number of Employees')


# In[235]:


for column in df.columns:
    if df[column].dtype == object:
        print(str(column) + ' : ' + str(df[column].unique()))
        print(df[column].value_counts())
        print('____________________________________________')
        print(df[column].value_counts(normalize=True) * 100)
        print('____________________________________________')


# In[143]:


df = df.drop('Gender', axis = 1)
df = df.drop('FamilyOppinionAboutTheJob', axis = 1)
df = df.drop('AvailabilityOfTransportNearTheResidence', axis = 1)
df = df.drop('MedicalTest', axis = 1)
df = df.drop('EPF_Number__t2o', axis = 1)


# In[144]:


df.corr()


# In[145]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), annot=True, fmt='.0%')


# In[209]:


#fig.set_facecolor('lightgrey')
# Data to plot
labels = 'Unmarried', 'Married', 'NoSpouse'
sizes = [594, 473, 20]
colors = ['#B2BA35', '#30A361', '#1A683B']
explode = (0, 0, 0)  # explode 1st slice

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors, startangle=140,autopct='%1.2f%%',textprops={'color':"black",'size':14})
plt.axis('equal')
plt.title("Marital Status Distribution", bbox={'facecolor':'0.9','pad':15})
plt.show()


# In[ ]:




