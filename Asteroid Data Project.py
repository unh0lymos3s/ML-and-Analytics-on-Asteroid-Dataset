#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import warnings 
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


warnings.filterwarnings('ignore')


# In[3]:


dataset = pd.read_csv(r"C:\Users\moosa\OneDrive\Desktop\asteroid.csv")


# In[4]:


df = dataset[:100000]


# In[5]:


df


# In[6]:


df.info()


# In[7]:


#selecting features that seem important
df_selected = ['neo','pha','H','diameter','albedo','diameter_sigma','e','a','q','i','tp','moid_ld','clas']


# In[8]:


df = df[df_selected]


# In[9]:


df.head()


# In[10]:


from sklearn.preprocessing import LabelEncoder


# In[11]:


encoder = LabelEncoder()
encoder.fit(df.neo)
df['neo'] = encoder.transform(df.neo)
df['pha'] = encoder.fit_transform(df.pha)
df['clas'] = encoder.fit_transform(df.clas)


# In[ ]:





# In[12]:


df


# In[13]:


df.describe


# In[14]:


df.head(434)


# In[15]:


df.isna().sum()


# In[16]:


df.dropna(inplace = True)


# In[17]:


plt.scatter(df['diameter'],df['H'])
plt.xlim(0,1000)
plt.ylim(0,50)
plt.xlabel('diameter')
plt.ylabel('Absolute Magnitude (H)')
plt.title('correlation between Luminosity and Diameter')

plt.show()


# In[18]:


plt.scatter(df['albedo'],df['H'])
plt.xlim(0,1)
plt.ylim(0,25)
plt.xlabel('Albedo (Reflectiveness)')
plt.ylabel('Absolute Magnitude (H)')
plt.title('correlation between Luminosity and Reflectiveness')

plt.show()


# In[19]:


plt.scatter(df['q'],df['H'])
plt.xlim(0,11)
plt.ylim(0,25)
plt.xlabel('Distance from Sun (q)')
plt.ylabel('Absolute Magnitude (H)')
plt.title('correlation between Luminosity and Shortest Distance from Sun')

plt.show()


# In[20]:


plt.scatter(df['q'],df['albedo'])
plt.title('correlation between Luminosity and Shortest Distance from Sun')
plt.xlabel('Distance from Sun (q)')
plt.ylabel('Reflectiveness(Albedo)')


# <h1><i>Clustering

# In[21]:


#df_n = df.drop('class', axis=1)


# In[22]:


from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, verbose=0, perplexity=1, random_state=1)
#df_no_name = df.drop('name', axis=1)
TSNE_df = tsne.fit_transform(df)
plt.scatter(TSNE_df[:,0],TSNE_df[:,1])


# In[66]:


tsne = TSNE(n_components=2, verbose=0, perplexity=50, random_state=1)

TSNE_df = tsne.fit_transform(df)
plt.scatter(TSNE_df[:,0],TSNE_df[:,1], c=data_with_clusters['Clusters'], cmap = "rainbow")


# In[24]:


from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(df)
PCA_df=pca.transform(df)
plt.scatter(PCA_df[:, 0], PCA_df[:,1])


# In[25]:


from sklearn.cluster import KMeans
wcss=[]
for i in range(1,20
              ):
  kmeans = KMeans(i)
  kmeans.fit(df)
  wcss_iter = kmeans.inertia_
  wcss.append(wcss_iter)

number_clusters = range(1,20)
plt.plot(number_clusters,wcss)
plt.title('The Elbow title')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')


# In[26]:


KMeans = KMeans(5)
KMeans.fit(df)


# In[27]:


identified_clusters = KMeans.fit_predict(df)
identified_clusters


# In[28]:


data_with_clusters = df.copy()
data_with_clusters['Clusters'] = identified_clusters
plt.scatter(df['diameter'],df['H'],c=data_with_clusters['Clusters'],cmap='rainbow')
plt.title('correlation between Luminosity and Diameter')


# In[29]:


data_with_clusters = df.copy()
data_with_clusters['Clusters'] = identified_clusters
plt.scatter(df['albedo'],df['H'],c=data_with_clusters['Clusters'],cmap='rainbow')
plt.title('correlation between Luminosity and Reflectiveness')


# In[30]:


data_with_clusters = df.copy()
data_with_clusters['Clusters'] = identified_clusters
plt.scatter(df['q'],df['H'],c=data_with_clusters['Clusters'],cmap='rainbow')
plt.title('correlation between Luminosity and Shortest Distance from Sun')


# In[31]:


data_with_clusters = df.copy()
data_with_clusters['Clusters'] = identified_clusters
plt.scatter(df['q'],df['albedo'],c=data_with_clusters['Clusters'],cmap='rainbow')
plt.title('correlation between Reflectiveness and Shortest Distance from Sun')


# <h1> <i>Regression<i>

# In[32]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split


# In[33]:


X = df.drop('diameter', axis = 1)
Y = df['diameter']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)


# In[34]:


X


# In[35]:


clf = LinearRegression()
clf.fit(X_train,Y_train)


# In[36]:


print("Linear Regression Train", clf.score(X_train,Y_train))
print("Linear Regression Test", clf.score(X_test,Y_test))


# In[37]:


clf = DecisionTreeRegressor()
clf.fit(X_train,Y_train)


# In[38]:


print("DecisionTree Train", clf.score(X_train,Y_train))
print("DecisionTree Test", clf.score(X_test,Y_test))


# In[39]:


clf = RandomForestRegressor()
clf.fit(X_train,Y_train)


# In[40]:


print("RandomForest Train", clf.score(X_train,Y_train))
print("RandomForest Test", clf.score(X_test,Y_test))


# In[41]:


clf = GradientBoostingRegressor()
clf.fit(X_train,Y_train)


# In[42]:


print("Gradient Boosting Train", clf.score(X_train,Y_train))
print("Gradient Boosting Test", clf.score(X_test,Y_test))


# <h1><i>Classification 

# In[43]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# In[44]:


x = df.drop('clas', axis = 1)
y = df['clas']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)


# In[45]:


clf = LogisticRegression()
clf.fit(x_train,y_train)


# In[46]:


print("Logistic Regression Train", clf.score(x_train,y_train))
print("Logistic Regression Test", clf.score(x_test,y_test))


# In[47]:


clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)


# In[48]:


print("DecisionTree Train", clf.score(x_train,y_train))
print("DecisionTree Test", clf.score(x_test,y_test))


# In[49]:


clf = RandomForestClassifier()
clf.fit(x_train,y_train)


# In[50]:


print("DecisionTree Train", clf.score(x_train,y_train))
print("DecisionTree Test", clf.score(x_test,y_test))


# In[51]:


clf = GradientBoostingClassifier(max_depth = 1)
clf.fit(x_train,y_train)


# In[52]:


print("GradientBoosting Train", clf.score(x_train,y_train))
print("GradientBoosting Test", clf.score(x_test,y_test))


# <h1><i>Anomaly Detection

# In[53]:


from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN


# In[54]:


clf = IsolationForest(contamination=0.07, random_state=42)
clf.fit(df)


# In[55]:


y_pred = clf.predict(df)


# In[56]:


df['anomaly'] = y_pred


# In[57]:


df


# In[58]:


plt.scatter(df[df['anomaly']==1]['albedo'], df[df['anomaly']==1]['H'], c='blue',label='inlier')
plt.scatter(df[df['anomaly']==-1]['albedo'], df[df['anomaly']==-1]['H'], c='red',label='outlier')
plt.title('Isolation Forest Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()


# In[59]:


svm = OneClassSVM(gamma='scale', nu=0.08)
svm.fit(df)


# In[60]:


y_pred_SVM = svm.predict(df)


# In[61]:


df['anomaly_svm'] = y_pred_SVM


# In[62]:


plt.scatter(df[df['anomaly_svm']==1]['albedo'], df[df['anomaly_svm']==1]['H'], c='blue',label='inlier')
plt.scatter(df[df['anomaly_svm']==-1]['albedo'], df[df['anomaly_svm']==-1]['H'], c='red',label='outlier')
plt.scatter(df[df['anomaly_svm']==0]['albedo'], df[df['anomaly_svm']==0]['H'], c='black',label='inlier2')

plt.title('Isolation Forest Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()


# In[63]:


dbscan = DBSCAN(eps=0.1)
clusters = dbscan.fit_predict(df)


# In[64]:


plt.scatter(df['albedo'], df['H'], c=clusters)
plt.show()

