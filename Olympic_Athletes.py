import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
athletes = pd.read_csv("C:\\Users\\khila\\OneDrive\\Documents\\Business Intelligence II\\Project\\Olympic_Athletes.csv")

# Preprocessing
athletes['weight'].fillna(athletes['weight'].median(), inplace=True)
athletes['height'].fillna(athletes['height'].median(), inplace=True)

# Feature Selection
features = athletes[['height', 'weight']]

# Standardization
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
athletes['Cluster'] = kmeans.fit_predict(features_scaled)

# Visualize the clusters
plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=athletes['Cluster'], cmap='viridis')
plt.xlabel('Height (scaled)')
plt.ylabel('Weight (scaled)')
plt.title('Athlete Clusters')
plt.show()

# Save the cluster results
athletes.to_csv('athlete_clusters.csv', index=False)