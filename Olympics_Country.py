import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
countries = pd.read_csv("C:\\Users\\khila\\OneDrive\\Documents\\Business Intelligence II\\Project\\Olympics_Country.csv")

# Check if the data is loaded correctly
print(countries.head())
print(countries.info())

# Encoding categorical data
features = pd.get_dummies(countries['country'])

# Standardization
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
countries['Cluster'] = kmeans.fit_predict(features_scaled)

# Visualize the clusters
plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=countries['Cluster'], cmap='viridis')
plt.xlabel('Feature 1 (scaled)')
plt.ylabel('Feature 2 (scaled)')
plt.title('Country Clusters')
plt.show()

# Save the cluster results
countries.to_csv('country_clusters.csv', index=False)
