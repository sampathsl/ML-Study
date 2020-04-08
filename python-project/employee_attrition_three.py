#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[2]:


df = pd.read_csv("New_Datset_Employee_Turnover_Old1.csv")
# df = pd.read_csv("New_Datset_Employee_Turnover_Original.csv")
# df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
# print(df)
# df.head()


# In[3]:


df.shape


# In[4]:


df.dtypes


# In[439]:


df.isna().sum()


# In[5]:


df.isnull().values.any()


# In[441]:


df['Is Terminated'].value_counts()


# In[7]:


df.describe()


# In[443]:


import matplotlib.pyplot as plt
plt.title('Distribution of Employee Termination')
plt.xlabel('Number of Axles')
sns.set(rc={'figure.figsize':(10,8)})
ax = sns.countplot(df['Is Terminated'])
ax.set_ylabel('Number of Employee')
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total * 100),
            ha="center")


# In[11]:


(1087 - 539) / 1087
total = df.shape[0]


# In[445]:


#Show the number of employees that left and stayed by age

# plt.subplots(figsize=(25,12))
# sns.countplot(x='Age Joined', hue='Is Terminated', data= df, palette = 'colorblind')
# sns.catplot('Age Joined', data=df, aspect=3, kind='count', hue='Is Terminated', palette=['C1', 'C0']).set_ylabels('Number of Employees')

plt.subplots(figsize=(20,8))
ax = sns.countplot(x='Age_Joined', hue='Is Terminated', data= df, palette = 'colorblind')
ax.set_xlabel('Age Joined')
ax.set_ylabel('Number of Employee')
ax.set_title("Distribution of Employee Termination Over Age Joined")


# In[12]:


#Show the number of employees that left and stayed by age
import matplotlib.pyplot as plt
#plt.subplots(figsize=(20,8))
#sns.countplot(x='Age Category', hue='Is Terminated', data= df, palette = 'colorblind')
# sns.catplot('Age_cat', data=df, aspect=3, kind='count', hue='Is Terminated', palette=['C1', 'C0']).set_ylabels('Number of Employees')

# plt.subplots(figsize=(10,8))
# ax = sns.countplot(x='Age_cat', hue='Is Terminated', data= df, palette = 'colorblind')
# ax.set_xlabel('Age Category')
# ax.set_ylabel('Number of Employee')
# ax.set_title("Distribution of Employee Termination Over Age Category")
# for p in ax.patches:
#     height = p.get_height()
#     ax.text(p.get_x()+p.get_width()/2.,
#             height + 3,
#             '{:1.2f}%'.format(height/total * 100),
#             ha="center")


plt.subplots(figsize=(10,8))
ax = sns.countplot(x='Age Category', hue='Is Terminated', data= df, palette = 'colorblind')
ax.set_xlabel('Age Category')
ax.set_ylabel('Number of Employee')
ax.set_title("Distribution of Employee Termination Over Age Category")
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total * 100),
            ha="center")


# In[8]:


for column in df.columns:
    if df[column].dtype == object:
        print(str(column) + ' : ' + str(df[column].unique()))
        print(df[column].value_counts())
        print('____________________________________________')
        print(df[column].value_counts(normalize=True) * 100)
        print('____________________________________________')


# In[448]:


df = df.drop('Gender', axis = 1)
df = df.drop('FamilyOppinionAboutTheJob', axis = 1)
df = df.drop('AvailabilityOfTransportNearTheResidence', axis = 1)
df = df.drop('MedicalTest', axis = 1)
df = df.drop('EPF_Number__t2o', axis = 1)


# In[432]:


df.corr()


# In[449]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), annot=True, fmt='.0%')


# In[450]:


df = df.drop('ExpectedSalary', axis = 1)

plt.figure(figsize=(20,20))
df.plot.box(df.corr())


# In[357]:


#fig.set_facecolor('lightgrey')
# Data to plot
labels = 'Unmarried', 'Married', 'NoSpouse'
sizes = [594, 473, 20]
colors = ['#B2BA35', '#30A361', '#1A683B']
explode = (0, 0, 0)  # explode 1st slice

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors, startangle=140,autopct='%1.2f%%',textprops={'color':"black",'size':14})
plt.axis('equal')
plt.title("Marital Status Distribution", bbox={'facecolor':'1','pad':15})
plt.show()


# In[383]:


#Show the number of employees that left and stayed by age

# g = sns.catplot('CivilStatus_cat', data=df, aspect=3, kind='count', hue='Is Terminated', palette=['C1', 'C0']).set_ylabels('Number of Employees')
# g.fig.suptitle('Distribution of Employee Termination')

plt.subplots(figsize=(10,6))
ax = sns.countplot(x='CivilStatus_cat', hue='Is Terminated', data= df, palette = 'colorblind')
ax.set_xlabel('Civil Status Category')
ax.set_ylabel('Number of Employee')
ax.set_title("Distribution of Employee Termination Over Civil Status")
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total * 100),
            ha="center")


# In[382]:


# g = sns.catplot('Recidence', data=df, aspect=3, kind='count', hue='Is Terminated', palette=['C1', 'C0']).set_ylabels('Number of Employees')
# g.fig.suptitle('Distribution of Employee Termination')

plt.subplots(figsize=(10,6))
ax = sns.countplot(x='Recidence', hue='Is Terminated', data= df, palette = 'colorblind')
ax.set_xlabel('Recidence')
ax.set_ylabel('Number of Employee')
ax.set_title("Distribution of Employee Termination Over Recidence")
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total * 100),
            ha="center")


# In[381]:


# g = sns.catplot('HighestEducationQualification_cat', data=df, aspect=3, kind='count', hue='Is Terminated', palette=['C1', 'C0']).set_ylabels('Number of Employees')
# g.fig.suptitle('Distribution of Employee Termination')

plt.subplots(figsize=(10,6))
ax = sns.countplot(x='HighestEducationQualification_cat', hue='Is Terminated', data= df, palette = 'colorblind')
ax.set_xlabel('Highest Education Qualification Category')
ax.set_ylabel('Number of Employee')
ax.set_title("Distribution of Employee Termination Over Highest Education Qualification")
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total * 100),
            ha="center")


# In[380]:


# g = sns.catplot('SpouseOccupation_cat', data=df, aspect=3, kind='count', hue='Is Terminated', palette=['C1', 'C0']).set_ylabels('Number of Employees')
# g.fig.suptitle('Distribution of Employee Termination')

plt.subplots(figsize=(10,6))
ax = sns.countplot(x='SpouseOccupation_cat', hue='Is Terminated', data= df, palette = 'colorblind')
ax.set_xlabel('Spouse Occupation Category')
ax.set_ylabel('Number of Employee')
ax.set_title("Distribution of Employee Termination Over Spouse Occupation")
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total * 100),
            ha="center")


# In[379]:


# g = sns.catplot('RetentionCategory', data=df, aspect=3, kind='count', hue='Is Terminated', palette=['C1', 'C0']).set_ylabels('Number of Employees')
# g.fig.suptitle('Distribution of Employee Termination')

plt.subplots(figsize=(10,6))
ax = sns.countplot(x='RetentionCategory', hue='Is Terminated', data= df, palette = 'colorblind')
ax.set_xlabel('Retention Category')
ax.set_ylabel('Number of Employee')
ax.set_title("Distribution of Employee Termination Over Retention Category")
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total * 100),
            ha="center")


# In[378]:


# g = sns.catplot('ContributionToTheFamilyIncome_cat', data=df, aspect=3, kind='count', hue='Is Terminated', palette=['C1', 'C0']).set_ylabels('Number of Employees')
# g.fig.suptitle('Distribution of Employee Termination')

plt.subplots(figsize=(10,6))
ax = sns.countplot(x='ContributionToTheFamilyIncome_cat', hue='Is Terminated', data= df, palette = 'colorblind')
ax.set_xlabel('Contribution To The Family Income Category')
ax.set_ylabel('Number of Employee')
ax.set_title("Distribution of Employee Termination Over Contribution To The Family Income")
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total * 100),
            ha="center")


# In[377]:


# g = sns.catplot('Children_cat', data=df, aspect=3, kind='count', hue='Is Terminated', palette=['C1', 'C0']).set_ylabels('Number of Employees')
# g.fig.suptitle('Distribution of Employee Termination')

plt.subplots(figsize=(10,6))
ax = sns.countplot(x='Children_cat', hue='Is Terminated', data= df, palette = 'colorblind')
ax.set_xlabel('Children Category')
ax.set_ylabel('Number of Employee')
ax.set_title("Distribution of Employee Termination Over Children Category")
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total * 100),
            ha="center")


# In[376]:


# g = sns.catplot('ApparelExperience', data=df, aspect=3, kind='count', hue='Is Terminated', palette=['C1', 'C0']).set_ylabels('Number of Employees')
# g.fig.suptitle('Distribution of Employee Termination')


plt.subplots(figsize=(10,6))
ax = sns.countplot(x='ApparelExperience', hue='Is Terminated', data= df, palette = 'colorblind')
ax.set_xlabel('Apparel Experience')
ax.set_ylabel('Number of Employee')
ax.set_title("Distribution of Employee Termination Over Apparel Experience")
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total * 100),
            ha="center")


# In[10]:


total = df.shape[0]
plt.subplots(figsize=(10,6))
ax = sns.countplot(x='PreviousWorkPlace', hue='Is Terminated', data= df, palette = 'colorblind')
ax.set_xlabel('Previous Work Place')
ax.set_ylabel('Number of Employee')
ax.set_title("Distribution of Employee Termination Over Previous Work Place")
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total * 100),
            ha="center")


# In[393]:


total = df.shape[0]
plt.subplots(figsize=(10,6))
ax = sns.countplot(x='ExperiencedSection_cat', hue='Is Terminated', data= df, palette = 'colorblind')
ax.set_xlabel('Experienced Sections')
ax.set_ylabel('Number of Employee')
ax.set_title("Distribution of Employee Termination Over Experienced Section")
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total * 100),
            ha="center")


# In[394]:


total = df.shape[0]
plt.subplots(figsize=(10,6))
ax = sns.countplot(x='ExpectationOfDoingTheJob_cat', hue='Is Terminated', data= df, palette = 'colorblind')
ax.set_xlabel('Expectation Of Doing The Job')
ax.set_ylabel('Number of Employee')
ax.set_title("Distribution of Employee Termination Over Expectation Of Doing The Job")
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total * 100),
            ha="center")


# In[396]:


total = df.shape[0]
plt.subplots(figsize=(20,6))
ax = sns.countplot(x='ReasonForChooseApparel', hue='Is Terminated', data= df, palette = 'colorblind')
ax.set_xlabel('Reason For Choose Apparel Category')
ax.set_ylabel('Number of Employee')
ax.set_title("Distribution of Employee Termination Over Reason For Choose Apparel")
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total * 100),
            ha="center")


# In[398]:


total = df.shape[0]
plt.subplots(figsize=(10,6))
ax = sns.countplot(x='PersonalImpression', hue='Is Terminated', data= df, palette = 'colorblind')
ax.set_xlabel('Personal Impression')
ax.set_ylabel('Number of Employee')
ax.set_title("Distribution of Employee Termination Over Personal Impression")
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total * 100),
            ha="center")


# In[406]:


total = df.shape[0]
plt.subplots(figsize=(10,8))
ax = sns.countplot(x='PermanentRecidence_cat', hue='Is Terminated', data= df, palette = 'colorblind')
ax.set_xlabel('Permanent Recidence')
ax.set_ylabel('Number of Employee')
ax.set_title("Distribution of Employee Termination Over Permanent Recidence")
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total * 100),
            ha="center")


# In[408]:


df.plot.hist(y='IQTestScore', bins=30);


# In[410]:


df.plot.box(y=['IQTestScore', 'Age_Joined'])


# In[ ]:




