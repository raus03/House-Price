# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 01:32:29 2020

@author: SUMIT GAURAV
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
dataset=pd.read_csv('train.csv')
import seaborn as sns
#Removing null values

dataset=dataset.drop(["Id","Alley","FireplaceQu","PoolQC","Fence","MiscFeature"],axis=1)

for i in dataset.columns:
    if dataset[i].dtype=='object':
        dataset[i]=np.where(dataset[i].isnull(),dataset[i].mode(),dataset[i])
    else:
        dataset[i]=np.where(dataset[i].isnull(),dataset[i].mean(),dataset[i])
dataset['MSSubClass']
#correlation
data2=dataset[num_feat].corr()
sns.heatmap(data2,annot=True)
plt.show()
data1=dataset.drop(['MSSubClass','LotArea','OverallCond','BsmtFinSF2','BsmtUnfSF','LowQualFinSF','BsmtFullBath','BsmtHalfBath','HalfBath','BedroomAbvGr','KitchenAbvGr','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold','GarageCars','GarageYrBlt','TotRmsAbvGrd','2ndFlrSF'],axis=1)    
#outliers
num_feat=[j for j in data1.columns if data1[j].dtype!="object"] 
outliers=[]
for i in num_feat:
    sns.scatterplot(data1[i],data1['SalePrice'],color='red')
    plt.title
    plt.show()
       
data1=data1.drop(data1.loc[data1['LotFrontage']>300].index,axis='index')

data1=data1.drop(data1.loc[data1['MasVnrArea']>1200].index,axis='index')
data1=data1.drop(data1.loc[data1['BsmtFinSF1']>5000].index,axis='index')
data1=data1.drop(data1.loc[data1['TotalBsmtSF']>6000].index,axis='index')
data1=data1.drop(data1.loc[data1['1stFlrSF']>4000].index,axis='index')
data1=data1.drop(data1.loc[data1['GrLivArea']>4500].index,axis='index')
data1=data1.drop(data1.loc[data1['GarageArea']>1200].index,axis='index')
data1=data1.drop(data1.loc[data1['OpenPorchSF']>400].index,axis='index')
len(data1.index)

#Printing categorial values
cat_feat=[j for j in data1.columns if data1[j].dtype=="object"]
#categorical_feature_masks=dataset.dtypes=='object'
#encoding categorical values
"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder=LabelEncoder()
for j in cat_feat:
    dataset[j]=label_encoder.fit_transform(dataset[j])
onehotencoder=OneHotEncoder(categorical_features=categorical_feature_masks,sparse=False)
dataset=onehotencoder.fit_transform(dataset)
dataset=pd.DataFrame(dataset)
dataset.drop(234,axis=1)

x=dataset.iloc[:,0:271]
y=dataset.iloc[:,-1]"""
 dummies=pd.get_dummies(data1[cat_feat],drop_first=True)
 data1=data1.drop(cat_feat,1)
 data1=pd.concat([data1,dummies],1)
 x=data1.drop('SalePrice',1)
 y=data1['SalePrice']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
#applying linear regression model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
#predicting the test set result
y_pred=regressor.predict(x_test)
regressor.score(x_test,y_test)
regressor.score(x_train,y_train)

#visualising the training set result
from sklearn.linear_model import Lasso
lasso=Lasso(alpha=100,random_state=0)
lasso.fit(x_train,y_train)
y_pred1=lasso.predict(x_test)
lasso.score(x_train,y_train)
lasso.score(x_test,y_test)

#Plotting residual plot
x_plot = plt.scatter(y_pred, (y_pred - y_test), c='b') 
plt.hlines(y=0, xmin= -1000, xmax=5000) 
plt.title('Residual plot')

#Finding outliers


    