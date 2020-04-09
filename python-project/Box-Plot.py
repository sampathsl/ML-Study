#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Employee_Turn_Over4.csv")
df.shape


# In[48]:


df.isna().sum()


# In[ ]:





# In[29]:


plt.figure(figsize=(12,12))
corr = df.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,
            cmap=sns.diverging_palette(220, 10, as_cmap=True), annot=True)


# In[26]:


g = sns.pairplot(df, hue="Is Terminated")


# In[12]:


g = sns.pairplot(df, hue="Is Terminated", palette="RdBu_r")


# In[11]:


g = sns.pairplot(df, vars=["Second Month Gross Salary","Second Month Basic Pay","Second Month OT","Second Month No Pay Days","Second Month Incentive"],hue="Is Terminated", palette="husl")


# In[13]:


g = sns.pairplot(df, vars=["Second Month Gross Salary","Second Month Basic Pay","Second Month OT","Second Month No Pay Days","Second Month Incentive"],hue="Is Terminated")


# In[21]:


# g = sns.FacetGrid(df, hue="Is Terminated", col="Second Month Gross Salary", height=4)
# g.map(qqplot, "Second Month Gross Salary")
# g.add_legend();

plt.figure(figsize=(30,50))
g = sns.pairplot(df, vars=["Second Month Gross Salary","Second Month Basic Pay","Second Month OT","Second Month No Pay Days","Second Month Incentive"],hue="Is Terminated", plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'})
# Title
plt.suptitle('Pair Plot of Employee Job Termination', size = 14);


# In[22]:


import seaborn as sns
sns.set(style="white")

# Load the example mpg dataset
mpg = sns.load_dataset("mpg")

# Plot miles per gallon against horsepower with other semantics
sns.relplot(x="horsepower", y="mpg", hue="origin", size="weight",
            sizes=(40, 400), alpha=.5, palette="muted",
            height=6, data=mpg)


# In[40]:


# plt.figure(figsize=(12,12))
# sns.scatterplot(x="Second Month Gross Salary", y="Salary_Category", hue="Is Terminated1",sizes=(40, 400), alpha=.5, palette="muted", data=df)

f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

# Add a graph in each part
sns.boxplot(df["Second Month Gross Salary"], ax=ax_box)
sns.distplot(df["Second Month Gross Salary"], ax=ax_hist)

# Remove x axis name for the boxplot
ax_box.set(xlabel='')


# In[ ]:


f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

# Add a graph in each part
sns.boxplot(df["Second Month Gross Salary"], ax=ax_box)
sns.distplot(df["Second Month Gross Salary"], ax=ax_hist)

# Remove x axis name for the boxplot
ax_box.set(xlabel='')


# In[49]:


g = sns.catplot(x="Is Terminated", y="Second Month Gross Salary",

                data=df, kind="box",

                height=10, aspect=.7);


# In[57]:


g = sns.catplot(x="Is Terminated", y="Second Month Basic Pay",

                data=df, kind="box",

                height=10, aspect=.5);


# In[58]:


g = sns.catplot(x="Is Terminated", y="Second Month Incentive",

                data=df, kind="box",

                height=10, aspect=.5);


# In[59]:


g = sns.catplot(x="Is Terminated", y="Second Month OT",

                data=df, kind="box",

                height=10, aspect=.5);


# In[ ]:


g = sns.catplot(x="Is Terminated", y="Second Month OT",

                data=df, kind="box",

                height=10, aspect=.5);
