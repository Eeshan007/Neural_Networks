#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers


# In[2]:


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


bank_df = pd.read_csv(r"bank-additional\bank-additional-full.csv")


# In[4]:


bank_df['job'] = bank_df['job'].replace('unknown',bank_df['job'].mode()[0])
bank_df['marital'] = bank_df['marital'].replace('unknown',bank_df['marital'].mode()[0])
bank_df['education'] = bank_df['education'].replace('unknown',bank_df['education'].mode()[0])
bank_df['default'] = bank_df['default'].replace('unknown',bank_df['default'].mode()[0])
bank_df['housing'] = bank_df['housing'].replace('unknown',bank_df['housing'].mode()[0])
bank_df['loan'] = bank_df['loan'].replace('unknown',bank_df['loan'].mode()[0])
bank_df['pdays'] = bank_df['pdays'].replace(999,28)


# In[5]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
bank_df["job"]=label_encoder.fit_transform(bank_df["job"])
bank_df["marital"]=label_encoder.fit_transform(bank_df["marital"])
bank_df["education"]=label_encoder.fit_transform(bank_df["education"])
bank_df["default"]=label_encoder.fit_transform(bank_df["default"])
bank_df["housing"]=label_encoder.fit_transform(bank_df["housing"])
bank_df["loan"]=label_encoder.fit_transform(bank_df["loan"])
bank_df["contact"]=label_encoder.fit_transform(bank_df["contact"])
bank_df["month"]=label_encoder.fit_transform(bank_df["month"])
bank_df["day_of_week"]=label_encoder.fit_transform(bank_df["day_of_week"])
bank_df["poutcome"]=label_encoder.fit_transform(bank_df["poutcome"])
bank_df["y"]=label_encoder.fit_transform(bank_df["y"])


# In[6]:


normalized_df = (bank_df.iloc[:,:20] - bank_df.iloc[:,:20].mean())/bank_df.iloc[:,:20].std()


# In[7]:


bank_df.iloc[:,:20] = normalized_df
bank_df


# In[8]:


X=bank_df.iloc[:,:20]
y=bank_df['y']


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


# # Neural Networks

# In[10]:


test_accuracy=[]


# ## Model 1

# In[11]:


model = Sequential()
model.add(Dense(40, input_dim=20, activation='relu'))
model.add(Dense(28, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[12]:


model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, verbose=0)


# In[13]:


_, accuracy = model.evaluate(X_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))


# In[14]:


predictions = model.predict_classes(X_test)


# In[15]:


_, accuracy1 = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy1*100))


# In[16]:


test_accuracy.append(accuracy1)


# ## Model 2

# In[17]:


model = Sequential()
model.add(Dense(50, input_dim=20, activation='sigmoid'))
model.add(Dense(39, activation='sigmoid'))
model.add(Dense(28, activation='sigmoid'))
model.add(Dense(17, activation='sigmoid'))
model.add(Dense(6, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))


# In[18]:


model.compile(loss='squared_hinge', optimizer='RMSprop', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, verbose=0)


# In[19]:


_, accuracy = model.evaluate(X_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))


# In[20]:


predictions = model.predict_classes(X_test)


# In[21]:


_, accuracy2 = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy2*100))


# In[22]:


test_accuracy.append(accuracy2)


# ## Model 3

# In[23]:


model = Sequential()
model.add(Dense(30, input_dim=20, activation='tanh'))
model.add(Dense(15, activation='tanh'))
model.add(Dense(4, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))


# In[24]:


model.compile(loss='hinge', optimizer='adam', metrics=['mean_squared_error','accuracy'])
history = model.fit(X_train, y_train, epochs=100, verbose=0)


# In[25]:


_,_, accuracy = model.evaluate(X_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))


# In[26]:


predictions = model.predict_classes(X_test)


# In[27]:


_,_, accuracy3 = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy3*100))


# In[28]:


test_accuracy.append(accuracy3)


# In[29]:


pyplot.plot(history.history['mean_squared_error'])
plt.show()
pyplot.plot(history.history['accuracy'],color='red')
plt.show()


# In[30]:


sns.lineplot([1,2,3],test_accuracy,color='red',label='Accuracy')


# # K Nearest Neighbors

# In[73]:


acc=[]
mse=[]
k=[]
for i in range(3,20,2):
    knn = KNeighborsClassifier(n_neighbors=i,metric='manhattan')
    knn.fit(X_train, y_train)
    y_pred=knn.predict(X_test)
    acc.append(accuracy_score(y_test,y_pred))
    mse.append(mean_squared_error(y_test,y_pred))
    k.append(i)


# In[74]:


sns.lineplot(k,acc,color='blue',label='Accuracy')
plt.show()
sns.lineplot(k,mse,color='blue',label='MSE')
plt.show()


# In[75]:


acc=[]
mse=[]
k=[]
for i in range(3,20,2):
    knn = KNeighborsClassifier(n_neighbors=i, metric='chebyshev')
    knn.fit(X_train, y_train)
    y_pred=knn.predict(X_test)
    acc.append(accuracy_score(y_test,y_pred))
    mse.append(mean_squared_error(y_test,y_pred))
    k.append(i)


# In[76]:


sns.lineplot(k,acc,color='blue',label='Accuracy')
plt.show()
sns.lineplot(k,mse,color='blue',label='MSE')
plt.show()


# In[77]:


acc=[]
k=[]
for i in range(3,20,2):
    knn = KNeighborsClassifier(n_neighbors=i, metric='euclidean')
    knn.fit(X_train, y_train)
    y_pred=knn.predict(X_test)
    acc.append(accuracy_score(y_test,y_pred))
    k.append(i)


# In[78]:


sns.lineplot(k,acc,color='blue',label='Accuracy')
plt.show()


# In[ ]:




