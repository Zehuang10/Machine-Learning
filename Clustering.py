# K Means Clustering with Python
## Import Libraries

import seaborn as sns
import matplotlib.pyplot as plt

## Create some Data

from sklearn.datasets import make_blobs

# Create Data
data = make_blobs(n_samples=200, n_features=2, 
                           centers=4, cluster_std=1.8,random_state=101)

## Visualize Data

plt.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')

## Creating the Clusters

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4)

kmeans.fit(data[0])

kmeans.cluster_centers_

kmeans.labels_

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_,cmap='rainbow')
ax2.set_title("Original")
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')

**Real Dataset Example**
import pandas as pd
import numpy as np
data = pd.read_csv('Mall_Customers.csv')
x = data.iloc[:,[3,4]].values

**Elbow Method**
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0) #n_init is the number of times k-means runs with different initial centroids
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')

**Draw a dendogram to find optimal number of clusters**
import scipy.cluster.hierarchy as sch
plt.figure(figsize=(10,5))
dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward')) #ward method is to minimize variance within each cluster
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')

**Fitting hierarchical clustering to the dataset**

from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean',linkage='ward')

y_hc = hc.fit_predict(x)

y_hc

**Visualize the clusters**

plt.scatter(x[y_hc == 0, 0], x[y_hc==0,1], s=100, c='red',label = 'Careful')
plt.scatter(x[y_hc == 1, 0], x[y_hc==1,1], s=100, c='blue',label = 'Standard')
plt.scatter(x[y_hc == 2, 0], x[y_hc==2,1], s=100, c='green',label = 'Target')
plt.scatter(x[y_hc == 3, 0], x[y_hc==3,1], s=100, c='cyan',label = 'Careless')
plt.scatter(x[y_hc == 4, 0], x[y_hc==4,1], s=100, c='magenta',label = 'Sensible')
plt.title('Clusters of Clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
