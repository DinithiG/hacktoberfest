from sklearn.manifold import TSNE

# Perform t-SNE to reduce dimensionality
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
data_tsne = tsne.fit_transform(data.drop('cluster', axis=1))

# Scatter plot using t-SNE components
plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=data['cluster'], cmap='viridis')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('K-Means Clustering (t-SNE)')
plt.show()
