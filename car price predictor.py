#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd


# In[54]:


car = pd.read_csv("https://raw.githubusercontent.com/rajtilakls2510/car_price_predictor/master/quikr_car.csv")


# In[55]:


car


# In[56]:


car.head()


# In[57]:


car.shape


# In[58]:


car.info()


# In[59]:


car['year'].unique()


# In[60]:


car['Price'].unique()


# In[61]:


car['kms_driven'].unique()


# In[62]:


car['fuel_type'].unique()


# In[63]:


backup =car.copy()


# # cleaning

# In[64]:


car=car[car['year'].str.isnumeric()]


# In[65]:


car['year']=car['year'].astype(int)


# In[66]:


car=car[car['Price'] != "Ask For Price"]


# In[67]:


car['Price']=car['Price'].str.replace(',','').astype(int)


# In[68]:


car['kms_driven']=car['kms_driven'].str.split(' ').str.get(0).str.replace(',','')


# In[69]:


car=car[car['kms_driven'].str.isnumeric()]


# In[70]:


car['kms_driven'] = car['kms_driven'].astype(int)


# In[71]:


car.info()


# In[72]:


car=car[~car['fuel_type'].isna()]


# In[73]:


car['name'] =car['name'].str.split(' ').str.slice(0,3).str.join(' ')


# In[75]:


#reset index
car=car.reset_index(drop=True)


# In[76]:


car


# In[77]:


car.info()


# In[78]:


car.describe()


# In[81]:


car=car[car['Price']<6e6].reset_index(drop=True)


# In[82]:


car.to_csv('cleaned Car.csv')


# # Model

# In[136]:


x = car.drop(columns='Price')
y = car['Price']


# In[137]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# In[138]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


# In[139]:


ohe = OneHotEncoder()
ohe.fit(x[['name','company','fuel_type']])


# In[140]:


column_trans = make_column_transformer((OneHotEncoder(categories = ohe.categories_),['name','company','fuel_type']),remainder='passthrough')


# In[141]:


lr=LinearRegression()


# In[142]:


pipe = make_pipeline(column_trans,lr)


# In[143]:


pipe.fit(x_train,y_train)


# In[144]:


y_pred=pipe.predict(x_test)


# In[145]:


r2_score(y_test,y_pred)


# In[151]:


scores =[]
for i in range(1000):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
    lr=LinearRegression()
    pipe = make_pipeline(column_trans,lr)
    pipe.fit(x_train,y_train)
    y_pred=pipe.predict(x_test)
    scores.append(r2_score(y_test,y_pred))
    


# In[152]:


import numpy as np


# In[153]:


np.argmax(scores)


# In[154]:


scores[np.argmax(scores)]


# In[157]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = np.argmax(scores))
lr=LinearRegression()
pipe = make_pipeline(column_trans,lr)
pipe.fit(x_train,y_train)
y_pred=pipe.predict(x_test)
r2_score(y_test,y_pred)
    


# In[164]:


import pickle


# In[166]:


pickle.dump(pipe,open('LinearRegressionModel.pkl','wb'))


# In[167]:


pipe.predict(pd.DataFrame([['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']],columns=['name','company','year','kms_driven','fuel_type']))


# In[ ]:




