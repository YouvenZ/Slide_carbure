  ## Channel Importance Clustering Analysis
  Understanding patterns in different action classes

---

# Problem Statement

- Need to understand how different channels contribute to model decisions
- Want to identify patterns across different action classes
- Goal: Cluster similar attribution patterns and visualize relationships

---

```json
CLASS_MAPPING = {
    0: 'color_rotation',
    1: 'color_temperature', 
    2: 'frequency_global',
    3: 'frequency_local',
    4: 'geometry',
    5: 'noise_global',
    6: 'noise_local',
    7: 'patch_endo_clone_adapted',
    8: 'patch_endo_clone_naive',
    9: 'patch_endo_clone_stamp',
    10: 'patch_endo_inpainting',
    11: 'patch_exo'
}
```


# Data Sources

- **Channel attributions**: Feature importance scores from Captum
- **Action metadata**: 12 distinct manipulation classes
  - Color manipulations (rotation, temperature)
  - Frequency manipulations (global, local)
  - Noise (global, local)
  - Various patch-based manipulations

---

# Analysis Pipeline

1. **Data Loading**: Extract embeddings and class labels
2. **Dimensionality Reduction**: PCA → t-SNE/UMAP
3. **Clustering**: K-means, DBSCAN, etc.
4. **Evaluation**: Silhouette score, cluster purity
5. **Visualization**: 2D plots with class annotations

---

# Dimensionality Reduction Options

<div grid="~ cols-2 gap-4">
<div>

### Linear Methods
- **PCA**: Principal Component Analysis
- Fast, interpretable, but linear

### Non-Linear Methods
- **t-SNE**: t-Distributed Stochastic Neighbor Embedding
- **UMAP**: Uniform Manifold Approximation and Projection
- Better for complex relationships

</div>
<div>

```python
# Code example
reducer = TSNE(
    n_components=2,
    perplexity=30,
    n_iter=1000,
    random_state=42
)
reduced_embeddings = reducer.fit_transform(
    scaled_embeddings
)
```

</div>
</div>

---

# Clustering Algorithms

<div grid="~ cols-2 gap-4">
<div>

### Partition-based
- **K-means**: Simple, efficient
- Requires number of clusters as input

### Hierarchical
- **Agglomerative**: Bottom-up approach

### Density-based
- **DBSCAN/HDBSCAN**: Handles noise, irregular shapes

</div>
<div>

```python
# Finding optimal clusters
for n_clusters in range(min_clusters, max_clusters + 1):
    kmeans = KMeans(
        n_clusters=n_clusters, 
        random_state=42
    )
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Evaluate with metrics
    silhouette_scores.append(
        silhouette_score(embeddings, cluster_labels)
    )
```

</div>
</div>

---
