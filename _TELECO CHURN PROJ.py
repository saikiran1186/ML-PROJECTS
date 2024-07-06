#!/usr/bin/env python
# coding: utf-8

# --- Customer Churn Prediction
#  Customer attrition or churn, is when customers stop doing business with in a company. It can have a significant impact on a company's revenue and it's crucial for businesses to find out the reasons why customers are leaving and take steps to reduce the number of customers leaving. One way to do this is by identifying customer segments that are at risk of leaving, and implementing retention strategies to keep them. Also, by using data and machine learning techniques, companies can predict which customers are likely to leave in the future and take actions to keep them before they decide to leave.
# 
# --- We are going to build a basic model for predicting customer churn using Telcommunication Customer Churn dataset

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split as split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


# In[3]:


df=pd.read_csv(r"C:\Users\saikiran\Downloads\churn_dataset.csv")
df


# In[3]:


df1=df.copy()
print(df1.columns)
len(df1.columns)


# In[11]:


sns.scatterplot(data=df, x='MonthlyCharges', y='TotalCharges', hue='Churn', style='Churn', alpha=0.6)
plt.title('Scatter Plot of Monthly Charges vs. Total Charges by Churn')
plt.xlabel('Monthly Charges ')
plt.ylabel('Total Charges ')
plt.show()


# In[33]:


plt.figure(figsize=(6,5))
data_to_plot = [df['tenure'], df['MonthlyCharges']]
plt.boxplot(data_to_plot, labels=['Tenure', 'Monthly Charges'], patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='black'),
            whiskerprops=dict(color='green'),
            capprops=dict(color='blue'),
            medianprops=dict(color='red'))
plt.title('Tenure and Monthly Charges')
plt.ylabel('Values')


# In[67]:


df["TotalCharges"] = df["TotalCharges"].replace(' ', np.nan)

# Convert the column to float
df["TotalCharges"] = df["TotalCharges"].astype('float64')

# For example, you can fill NaN values with the mean of the column
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())


# In[69]:


df.info()


# In[72]:


plt.figure(figsize=(6,5))
sns.boxplot(df['TotalCharges'])
plt.title('Distribution of Total Charges')  # Adds a title to the plot
plt.ylabel('Total Charges')  # Label for the y-axis
plt.show()


# In[34]:


df1["tenure"].median()


# ## Exploratory Data Analysis
# Problem Statement - Given various features about a customer like Gender, SeniorCitizen, Partner, Dependents etc.. , predict if the customer will churn or not.

# In[4]:


dup_rows=df1[df1.duplicated()]
dup_rows


# In[ ]:





# In[8]:


sns.countplot(data=df1,x="gender",hue="Churn")


# ### observation:
# ### from the above countplot we can observe that almost both genders have churned equally
# ### but male customers churned less compared to female

# In[79]:


sns.countplot(data=df1,x="Dependents",hue="Churn",color="yellow",palette="viridis")


# In[73]:


sns.countplot(data=df1,x="Partner",hue="Churn",color="blue",palette="rocket")


# ### observation:
# ### from the graph we can see that customers who have no partners churned higher than 
# ### the customers who have

# In[ ]:





# In[ ]:





# In[80]:


sns.histplot(data=df1,x="tenure",multiple="stack",hue=df["Churn"])


# ### From above histplot we can say that people who have less tenure tend to churn more compared to those who have more tenure

# In[81]:


sns.histplot(data=df1,x="Contract",multiple="stack",hue=df1["Churn"],color="yellow",palette="viridis")


# ### obs:
# ### Individuals on month-to-month contracts are more prone to churn compared to those who are on annual or longer-term contracts

# In[85]:


sns.histplot(data=df1,x="PaperlessBilling",multiple="stack",hue=df1["Churn"],color="yellow",palette="plasma")


# ### obs:
# ### Customers with paperless billing have higher churn rates

# In[15]:


sns.histplot(data=df1,x="PhoneService",multiple="stack",hue=df1["Churn"])


# ### People with phoneservice tend to churn more compared to those who have no phone service

# In[17]:


sns.histplot(data=df1,x="InternetService",multiple="stack",hue=df1["Churn"])


# ### The fiber optic internet service customers tend to churn more compared to DSL and no internet service

# In[18]:


sns.histplot(data=df1,x="OnlineSecurity",multiple="stack",hue=df1["Churn"])


# ### people with no onlinesecurity tend to churn more than those with online security and no internet service

# In[19]:


sns.histplot(data=df1,x="OnlineBackup",multiple="stack",hue=df1["Churn"])


# ## Observation
# ## From the stack-bar chart above, it seems that most customers churned who did not opted onlinebackup services

# In[20]:


sns.histplot(data=df1,x="StreamingTV",multiple="stack",hue=df1["Churn"])


# ## Observation
# ## From the stack-bar chart above, it appears that those who churn most are those who have not opted any streaming tv services

# In[76]:


import matplotlib.pyplot as plt
sns.histplot(data=df1, x="PaymentMethod", multiple="stack", hue=df1["Churn"])
plt.xticks(rotation=45) 
plt.show()


# ## Observation:
# ## From the stack-bar chart above, it appears that those who churn most are those with electronic check as a payment method.

# In[ ]:





# In[43]:


df1.drop(columns=["customerID"],axis=1,inplace=True)


# In[44]:


df1


# In[45]:


#step-1
#- Identifing the input and output/target variables
y=df1.iloc[:,-1]#target variable
x=df1.iloc[:,:19]

#step-2
#- Identify the type of ML Task.
Because the target variable is present in historical data and we can 
give the input and output variables to model it is "supervised learning". 
#- since the target or dependent variable is categorical with nominal in nature 
we can use "classification" task.#step-3
The evaluation metric for classification task would be "Accuracy"
# In[46]:


#Step - 4: Spliting the dataset into Training and Testing (75:25 split)

x_train,x_test,y_train,y_test=split(x,y,train_size=0.75,random_state=42)


# In[47]:


print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

#Step - 5: Data preparation on train data:
- For Numerical Variables - Standardization or Normalization (Fit and Transform)
- For Categorical - LabelEncoding or OneHotEncoding
# In[53]:


x_train.info()


# ### observation:
# ### from the info we can see that there are 5282 entries and no null values and we have 15 objects,2 float and 2 int

# In[49]:


#we need to change the datatype of "Totalcharges" because it is in strings and cantains empty string
x_train["TotalCharges"].value_counts()


# In[50]:


# Replace empty strings with NaN
x_train["TotalCharges"] = x_train["TotalCharges"].replace(' ', np.nan)

# Convert the column to float
x_train["TotalCharges"] = x_train["TotalCharges"].astype('float64')

# For example, you can fill NaN values with the mean of the column
x_train["TotalCharges"] = x_train["TotalCharges"].fillna(x_train["TotalCharges"].median())


# In[51]:


# Replace empty strings with NaN in x_test
x_test["TotalCharges"] = x_test["TotalCharges"].replace(' ', np.nan)

# Convert the column to float in x_test
x_test["TotalCharges"] = x_test["TotalCharges"].astype('float64')

# For example, you can fill NaN values with the mean of the column
x_test["TotalCharges"] = x_test["TotalCharges"].fillna(x_test["TotalCharges"].median())


# In[33]:


x_train_cat=x_train.select_dtypes("object")
x_train_num=x_train.select_dtypes(include=["int64","float64"])


# In[34]:


#checking if there are null values
x_train_cat.isnull().sum()


# ### In training category we have no null values in any  column

# In[35]:


#categorical to numerical transformation
ohe = OneHotEncoder(drop="first", sparse_output=False)
x_train_cat_encoded = ohe.fit_transform(x_train_cat)
x_train_cat_one = pd.DataFrame(x_train_cat_encoded, 
                               columns=ohe.get_feature_names_out(x_train_cat.columns),
                               index=x_train_cat.index)
x_train_cat_one


# In[ ]:





# In[36]:


x_train_num


# In[37]:


#standardizing numerical columns
stand=StandardScaler()
x_train_num_rescaled=pd.DataFrame(stand.fit_transform(x_train_num),
                                 columns=stand.get_feature_names_out(x_train_num.columns),
                                 index=x_train_num.index)
x_train_num_rescaled


# In[38]:


#x_train_transformed
x_train_transformed=pd.concat([x_train_cat_one,x_train_num_rescaled],axis=1)
x_train_transformed


# In[39]:


x_test_cat=x_test.select_dtypes("object")
x_test_num=x_test.select_dtypes(include=["int64","float64"])


# In[40]:


x_test_cat


# In[42]:


#categorical to numerical transformation for x_test
ohe = OneHotEncoder(drop="first", sparse_output=False)
ohe.fit(x_train_cat)
x_test_cat_encoded = ohe.transform(x_test_cat)
x_test_cat_one = pd.DataFrame(x_test_cat_encoded, 
                               columns=ohe.get_feature_names_out(x_test_cat.columns),
                               index=x_test_cat.index)
x_test_cat_one


# In[43]:


#standardizing numerical columns for x_test
stand=StandardScaler()
stand.fit(x_train_num)
x_test_num_rescaled=pd.DataFrame(stand.transform(x_test_num),
                                 columns=stand.get_feature_names_out(x_test_num.columns),
                                 index=x_test_num.index)
x_test_num_rescaled


# In[44]:


#x_test_transformed
x_test_transformed=pd.concat([x_test_cat_one,x_test_num_rescaled],axis=1)
x_test_transformed


# In[45]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[46]:


# using Logistic Regression
from sklearn.linear_model import LogisticRegression
LR_classifier = LogisticRegression()
LR_classifier.fit(x_train_transformed,y_train)


# In[47]:


y_test_pred_LR=LR_classifier.predict(x_test_transformed)


# In[48]:


metrics.accuracy_score(y_test,y_test_pred_LR)


# In[49]:


sns.histplot(y_test,color="blue",alpha=0.5)
sns.histplot(y_test_pred_LR,color="red",alpha=0.5)


# In[50]:


#DECISION TREE CLASSIFIER
from sklearn.tree import DecisionTreeClassifier

DT_classifier = DecisionTreeClassifier()

DT_classifier.fit(x_train_transformed,y_train)


# In[51]:


y_test_pred_DT=DT_classifier.predict(x_test_transformed)


# In[52]:


y_test_pred_DT


# In[53]:


metrics.accuracy_score(y_test,y_test_pred_DT)


# In[54]:


sns.histplot(y_test,color="yellow",alpha=0.5)
sns.histplot(y_test_pred_DT,color="red",alpha=0.5)


# In[55]:


from sklearn.svm import SVC

svm_classifier = SVC()

svm_classifier.fit(x_train_transformed,y_train)


# In[56]:


y_test_pred_svm=svm_classifier.predict(x_test_transformed)


# In[57]:


y_test_pred_svm


# In[58]:


metrics.accuracy_score(y_test,y_test_pred_svm)


# In[59]:


sns.histplot(y_test,color="yellow",alpha=0.5)
sns.histplot(y_test_pred_svm,color="red",alpha=0.5)


# In[60]:


from sklearn.ensemble import RandomForestClassifier

RF_classifier = RandomForestClassifier()

RF_classifier.fit(x_train_transformed,y_train)


# In[61]:


y_test_pred_RF=RF_classifier.predict(x_test_transformed)


# In[62]:


y_test_pred_RF


# In[63]:


metrics.accuracy_score(y_test,y_test_pred_RF)


# In[64]:


sns.histplot(y_test,color="yellow",alpha=0.5)
sns.histplot(y_test_pred_RF,color="pink",alpha=0.5)


# In[65]:


# KNN CLASSIFIER
from sklearn.neighbors import KNeighborsClassifier

KNN_classifier = KNeighborsClassifier()

KNN_classifier.fit(x_train_transformed,y_train)


# In[66]:


y_test_pred_knn = KNN_classifier.predict(x_test_transformed)


# In[67]:


metrics.accuracy_score(y_test,y_test_pred_knn)


# In[69]:


sns.histplot(y_test,color="blue",alpha=0.5)
sns.histplot(y_test_pred_DT,color="black",alpha=0.5)


# In[70]:


d={
    "Algo":["Logistic Regression","DecisionTreeClassifier"," SVC","RandomForestClassifier","KNeighborsClassifier"],
    "Accuracy_score":[0.8126064735945485,0.7257240204429302,0.80465644520159,0.7876206700738216,0.7666098807495741]
}
pd.DataFrame(d)


# ## Insights
Based on the Accuracy metric, the best model for the classification problem appears to be Logistic Regression model. Although all the other models had similar accuracy scores, Logistic Regression model had the highest Accuracy. 
While Logistic Regression model had a good accuracy score of 0.81, Therefore, based on the metrics evaluated Logistic Regression model appears to be the best model for this classification problem.
# In[ ]:




