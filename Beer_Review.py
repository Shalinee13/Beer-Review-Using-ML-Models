#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries and Dataset:

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


data=pd.read_csv("Beer.csv")
data


# # Exploratory Data Analysis (EDA):

# In[4]:


data.shape


# ###### Observations:
# 
# Data set contains 37500 rows and 19 columns.

# In[5]:


data.columns


# # Top 5 values

# In[6]:


data.head()


# # last 5 values

# In[7]:


data.tail()


# # Finding the information of the data:

# In[8]:


data.info()


# # Finding the Null Values in the Dataset:¶

# In[9]:


data.isnull()


# In[10]:


data.isnull().sum()


# In[11]:


data_new = data.dropna(subset=['review_text', 'user_profileName'])


# In[12]:


data_new.isnull().sum()


# In[13]:


data_new.drop(['user_ageInSeconds','user_birthdayRaw','user_birthdayUnix','user_gender'], axis=1, inplace=True)


# In[14]:


data_new.isnull().sum()


# In[15]:


data_new.drop(['index','review/timeStruct','review_timeUnix'], axis=1, inplace=True)


# In[16]:


data_new.duplicated().sum()


# In[17]:


data_new.shape


# # Basic statistics:¶

# In[18]:


data_new.describe()


# In[19]:


sns.catplot(x="review_overall",kind="boxen",data=data_new)   ##box graph
plt.show()


# # Value count of columns:¶

# In[20]:


for i in data_new.columns:
  print(i, len(data_new[i].value_counts().index))


# In[21]:


aroma_r = data_new['review_aroma'].value_counts()


# In[22]:


aroma_r


# In[23]:


overall_r=data_new['review_overall'].value_counts()
overall_r.to_frame()


# # Unique values of every columns:

# In[24]:


data_new.nunique()


# # Data Insights:

# From the given data we can briefly infer about the different features as follows:
# 
# ##### beer_ABV : Alcohol by volume content of a beer
# ##### beer_beerId : Unique ID for beer identification
# ##### beer_brewerId : Unique ID identifying the brewer
# ##### beer_name : Name of the beer
# ##### beer_style : Beer Category
# ##### review_appearance: Rating based on how the beer looks [Range : 1-5]
# ##### review_palatte : Rating based on how the beer interacts with the palate [Range : 1-5]
# ##### review_overall : Overall experience of the beer is combined in this rating [Range : 1-5]
# ##### review_taste : Rating based on how the beer actually tastes [Range : 1-5]
# ##### review_profileName: Reviewer’s profile name / user ID
# ##### review_aroma : Rating based on how the beer smells [Range : 1-5]
# ##### review_text : Review comments/observations in text format
# ##### review_time : Time in UNIX format when review was recorded

# In[25]:


overall_r=pd.DataFrame(data_new['review_overall'].value_counts())
overall_r


# # Data Visualisation:

# In[26]:


plt.pie(overall_r["review_overall"],labels=['4','4.5','3.5','3','5','2.5','2','1.5','1','0'],autopct='%.1f%%')
plt.show()


# In[27]:


data_new.loc[:,['beer_name','beer_ABV']]


# In[28]:


data_new['beer_name'].value_counts().head(50).plot.bar(figsize=(16,5),title= 'Most Poular Beers by Name')
plt.show()


# In[ ]:





# In[29]:


data_new['beer_style'].value_counts().head(50).plot.bar(figsize=(16,5),title= 'Most Poular Beers by Style')
plt.show()


# In[ ]:





# In[30]:


plt.figure(figsize=(12,5))
sns.distplot(data_new['beer_ABV'],bins = 50)
plt.xlabel("Alcohol By Volume")
plt.show()


# Observation :
# 
# It can be infered that almost all of the majority data in the distribution of 'beer_ABV' is between 5-10 with long tail towards right.

# In[31]:


sns.catplot(x="beer_ABV",kind="boxen",data=data_new)   ##box graph
plt.show()


# In[32]:


data_new.hist(bins = 15,figsize=(16,12))
plt.show()


# In[ ]:





# In[33]:


data_rtro = data_new.loc[:,['beer_style','review_text']].sort_values('beer_style')
data_rtro


# In[34]:


sns.scatterplot(x='review_overall', y='brewerId', data=data_new)
plt.show()


# In[35]:


sns.pairplot(data_new)
plt.show()


# # Correlation:

# In[36]:


data_new.corr()


# In[37]:


plt.figure(figsize = (15,10))
sns.heatmap(data_new.corr(), annot=True)

plt.show()


# In[38]:


data_new


# In[39]:


data_new.drop(['beer_name','beer_style','review_text','user_profileName'], axis=1, inplace=True)


# ### Putting feature variables into X

# In[40]:


X=data_new.drop(["review_overall"], axis = 1)


# ### Putting target variable to y

# In[41]:


Y=data_new["review_overall"]


# # Import library

# In[42]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# ## Splitting the Dataset into Train & Test:

# In[43]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


# In[44]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# ## Linear Regression

# In[45]:


model1=LinearRegression()


# In[46]:


model1.fit(X_train,Y_train)


# In[47]:


y_pred=model1.predict(X_test)


# In[48]:


model1.score(X_train,Y_train)*100   


# In[49]:


model1.score(X_test,Y_test)*100


# In[50]:


new_df=pd.DataFrame({'Actual':Y_test,'Predicted':y_pred})


# In[51]:


new_df


# # Accuracy

# In[52]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# In[53]:


accuracy1=r2_score(Y_test, y_pred)*100
print('\nAccuracy for Linear Regression =',accuracy1)
print('\nMean Absolute error for Linear Regression =',mean_absolute_error(Y_test, y_pred))
print('\nMean Square error for Linear Regression =',mean_squared_error(Y_test, y_pred))


# # Decision Tree Regressor

# In[ ]:





# In[54]:


model=DecisionTreeRegressor()


# In[55]:


model.fit(X_train,Y_train)


# In[56]:


y_predict = model.predict(X_test)


# In[57]:


y_predict


# In[58]:


model.score(X_train,Y_train)*100


# In[59]:


model.score(X_test,Y_test)*100


# # Accuracy

# In[60]:


accuracy2=r2_score(Y_test, y_predict)*100
print('\nAccuracy for DecisionTree Regression =',accuracy2)
print('\nMean Absolute error for DecisionTree Regression =',mean_absolute_error(Y_test, y_predict))
print('\nMean Square error for DecisionTree Regression =',mean_squared_error(Y_test, y_predict))


# In[ ]:





# # Random Forest Regressor

# In[61]:


rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)


# In[62]:


rf_regressor.fit(X_train, Y_train)


# In[63]:


y_pred = rf_regressor.predict(X_test)


# In[64]:


rf_regressor.score(X_train,Y_train)*100


# In[65]:


rf_regressor.score(X_test,Y_test)*100


# In[ ]:





# # Accuracy

# In[66]:


accuracy3=r2_score(Y_test, y_pred)*100
print('\nAccuracy for RandomForestRegressor =',accuracy3)
print('\nMean Absolute error for RandomForestRegressor =',mean_absolute_error(Y_test, y_pred))
print('\nMean Square error for RandomForestRegressor =',mean_squared_error(Y_test, y_pred))


# In[ ]:





# # Comparing Accuracy of All Models:

# In[67]:


print('\nAccuracy for Linear Regression =',accuracy1)
print('\nAccuracy for DecisionTree Regression =',accuracy2)
print('\nAccuracy for RandomForestRegressor =',accuracy3)


# In[ ]:





# In[ ]:




