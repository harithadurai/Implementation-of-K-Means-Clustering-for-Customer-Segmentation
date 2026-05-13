# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the customer dataset.
2. Select the required features and determine the number of clusters using the Elbow Method.
3. Apply the K-Means clustering algorithm to group customers into clusters.
4. Visualize the clusters and display the segmented customer groups.
   

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: D.HARITHA.
RegisterNumber: 212225040118
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("D:\DATA\Mall_Customers.csv")

print("Customer Data:\n")
print(data.head())

X = data.iloc[:, [3, 4]].values

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++',
                    max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6,5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++',
                max_iter=300, n_init=10, random_state=0)

y_kmeans = kmeans.fit_predict(X)

plt.figure(figsize=(8,6))

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1],
            s=100, c='red', label='Cluster 1')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1],
            s=100, c='blue', label='Cluster 2')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1],
            s=100, c='green', label='Cluster 3')

plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1],
            s=100, c='cyan', label='Cluster 4')

plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1],
            s=100, c='magenta', label='Cluster 5')

plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=300, c='yellow', label='Centroids')

plt.title('Customer Segmentation using K-Means')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

print("\nK Means Clustering for Customer Segmentation completed successfully.")
```

## Output:

<img width="840" height="195" alt="image" src="https://github.com/user-attachments/assets/b54e42fc-a75c-4b81-805a-fd89fef144e6" />

<img width="947" height="597" alt="image" src="https://github.com/user-attachments/assets/2ee2b7d7-7d68-4da4-b718-e3b7aaae5b79" />


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
