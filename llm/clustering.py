import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import pandas as pd
from sklearn.decomposition import PCA
import json
import torch
from mpl_toolkits import mplot3d

def load_csv(filename):
    data = pd.read_csv(filename)
    embeddings_ = []
    scenes = data['scenes']
    embeddings = data['ada_embedding_sce']
    for emb in embeddings:
        emb = np.array(eval(emb))
        embeddings_.append(emb)
    embeddings_ = np.array(embeddings_)
    return scenes, embeddings_

def load_matched_sce_per(filename):
    with open(filename) as f:
        matched_data = json.load(f)
        selected_scenes = matched_data.keys()
    return selected_scenes

def load_mistral_embedding(filename):
    data = torch.load(filename)

    return data

# fake data

# np.random.seed(0)
# X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

scenes, embeddings = load_csv('./embedding/ada_scene_embedding.csv')
# selected_scenes = load_matched_sce_per('./embedding/match_sce_per_v4.json')
mistral_embedding_scene_1 = load_mistral_embedding('./embedding/mistral_embedding_scene_1.pth')
mistral_embedding_scene_2 = load_mistral_embedding('./embedding/mistral_embedding_scene_2.pth')
mistral_embedding_scene = (mistral_embedding_scene_1 + mistral_embedding_scene_2)/2.0
embeddings = mistral_embedding_scene

# K-means
kmeans = KMeans(n_clusters=5)
kmeans.fit(embeddings)
y_kmeans = kmeans.predict(embeddings)


# decomposition
model = PCA(n_components=3)
lower_dim_embedding = model.fit_transform(embeddings)

# plot
# plt.scatter(lower_dim_embedding[:, 0], lower_dim_embedding[:, 1], c=y_kmeans, s=50, cmap='viridis')
# # centers = kmeans.cluster_centers_
# # plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
# plt.title("K-means Clustering")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.show()
print('done!')


# plot a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# plt.scatter(lower_dim_embedding[:, 0], lower_dim_embedding[:, 1], lower_dim_embedding[:,2], c=y_kmeans, cmap='viridis')
ax.scatter3D(lower_dim_embedding[:, 0], lower_dim_embedding[:,1], lower_dim_embedding[:,2], c=y_kmeans, cmap='viridis')
plt.title("K-means Clustering")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
print('done!')