import numpy as np
import matplotlib.pyplot as plt

def k_means(X, k, max_iters=100):
    # Randomly initialize centroids
    centroids = X[np.random.choice(range(len(X)), k, replace=False)]

    for _ in range(max_iters):
        # Assign each data point to the nearest centroid
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)

        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    # Calculate the cost
    cost = np.sum(np.min(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1))

    return centroids, labels, cost

# Example data
X = np.array([[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]])

# Number of clusters
k = 2

# Perform K-means clustering
centroids, labels, cost = k_means(X, k)

# Plot the final clusters
plt.figure(figsize=(8, 6))
for i in range(k):
    cluster_points = X[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i+1}')

plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', label='Centroids')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('K-means Clustering')
plt.legend()
plt.grid(True)
plt.show()

print("Final Centroids:")
print(centroids)
print("\nCost:", cost)
