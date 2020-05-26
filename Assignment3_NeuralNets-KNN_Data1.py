#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers


# In[33]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot
from statistics import mean
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


# # Data Preparation

# In[3]:


project_df = pd.read_csv("sgemm_product_dataset\sgemm_product.csv")


# In[4]:


project_df['Run_Avg'] = project_df.iloc[:,14:18].mean(axis=1)


# In[5]:


project_df=project_df.drop(columns=['Run1 (ms)','Run2 (ms)','Run3 (ms)','Run4 (ms)'])


# In[6]:


project_df=project_df.dropna()


# In[7]:


project_df['Run_Avg'].median()
project_df['Run_Avg'] = np.where(project_df['Run_Avg'] >= project_df['Run_Avg'].median(), 1, 0)
#Converted all the values above median to 1 and below median to zero


# In[8]:


normalized_df = (project_df.iloc[:,:14] - project_df.iloc[:,:14].mean())/project_df.iloc[:,:14].std()


# In[9]:


project_df.iloc[:,:14] = normalized_df
project_df


# In[10]:


X=project_df.iloc[:,:14]
y=project_df['Run_Avg']


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


# # Neural Networks

# In[12]:


test_accuracy=[]


# ## Model 1

# In[13]:


model = Sequential()
model.add(Dense(25, input_dim=14, activation='sigmoid'))
model.add(Dense(18, activation='sigmoid'))
model.add(Dense(11, activation='sigmoid'))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))


# In[14]:


model.compile(loss='squared_hinge', optimizer='sgd', metrics=['accuracy'])


# In[15]:


model.fit(X_train, y_train, epochs=100, verbose=0)


# In[16]:


_, accuracy = model.evaluate(X_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))


# In[17]:


predictions = model.predict_classes(X_test)


# In[18]:


_, accuracy1 = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy1*100))


# In[19]:


test_accuracy.append(accuracy1)


# ## Model 2

# In[20]:


model = Sequential()
model.add(Dense(50, input_dim=14, activation='sigmoid'))
model.add(Dense(39, activation='tanh'))
model.add(Dense(28, activation='tanh'))
model.add(Dense(17, activation='tanh'))
model.add(Dense(6, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))


# In[21]:


model.compile(loss='hinge', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, verbose=0)


# In[22]:


_, accuracy = model.evaluate(X_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))


# In[23]:


predictions = model.predict_classes(X_test)


# In[24]:


_, accuracy2 = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy2*100))


# In[25]:


test_accuracy.append(accuracy2)


# ## Model 3

# In[26]:


model = Sequential()
model.add(Dense(30, input_dim=14, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[27]:


model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['mean_squared_error','accuracy'])
history = model.fit(X_train, y_train, epochs=100, verbose=0)


# In[28]:


_,_, accuracy = model.evaluate(X_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))


# In[29]:


predictions = model.predict_classes(X_test)


# In[30]:


_,_, accuracy3 = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy3*100))


# In[31]:


test_accuracy.append(accuracy3)


# In[35]:


pyplot.plot(history.history['mean_squared_error'])
plt.show()
pyplot.plot(history.history['accuracy'],color='red')
plt.show()


# In[36]:


sns.lineplot([1,2,3],test_accuracy,color='red',label='Accuracy')


# # K Nearest Neighbors

# In[66]:


acc=[]
mse=[]
k=[]
for i in range(3,12,2):
    knn = KNeighborsClassifier(n_neighbors=i,metric='manhattan')
    knn.fit(X_train, y_train)
    y_pred=knn.predict(X_test)
    acc.append(accuracy_score(y_test,y_pred))
    mse.append(mean_squared_error(y_test,y_pred))
    k.append(i)


# In[67]:


sns.lineplot(k,acc,color='blue',label='Accuracy')
plt.show()
sns.lineplot(k,mse,color='blue',label='MSE')
plt.show()


# In[46]:


acc=[]
k=[]
for i in range(3,12,2):
    knn = KNeighborsClassifier(n_neighbors=i,metric='euclidean')
    knn.fit(X_train, y_train)
    y_pred=knn.predict(X_test)
    acc.append(accuracy_score(y_test,y_pred))
    k.append(i)


# In[47]:


sns.lineplot(k,acc,color='blue',label='Accuracy')
plt.show()


# In[52]:


acc=[]
k=[]
for i in range(3,12,2):
    knn = KNeighborsClassifier(n_neighbors=i,metric='chebyshev')
    knn.fit(X_train, y_train)
    y_pred=knn.predict(X_test)
    acc.append(accuracy_score(y_test,y_pred))
    k.append(i)


# In[53]:


sns.lineplot(k,acc,color='blue',label='Accuracy')
plt.show()


# In[ ]:




