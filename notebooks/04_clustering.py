import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ── Load data ────────────────────────────────────────────────
X_scaled = pd.read_csv('data/cleaned/features_scaled.csv')
df = pd.read_csv('data/cleaned/steam_clean.csv')

# ── Elbow method ─────────────────────────────────────────────
inertias = []
silhouettes = []
K_range = range(2, 10)

print("Testing K values...")
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
    print(f"  K={k} | Inertia: {km.inertia_:,.0f} | Silhouette: {silhouette_score(X_scaled, labels):.4f}")

# ── Plot elbow + silhouette ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(K_range, inertias, marker='o', color='#534AB7', linewidth=2)
axes[0].set_title('Elbow method')
axes[0].set_xlabel('Number of clusters (K)')
axes[0].set_ylabel('Inertia')

axes[1].plot(K_range, silhouettes, marker='o', color='#1D9E75', linewidth=2)
axes[1].set_title('Silhouette score')
axes[1].set_xlabel('Number of clusters (K)')
axes[1].set_ylabel('Score (higher = better)')

plt.tight_layout()
plt.savefig('outputs/04_elbow_silhouette.png', dpi=150)
plt.show()
print("\nElbow chart saved.")

# ── Fit final model ───────────────────────────────────────────
# We'll start with K=4, you can adjust after seeing the elbow chart
BEST_K = 5
km_final = KMeans(n_clusters=BEST_K, random_state=42, n_init=10)
df['cluster'] = km_final.fit_predict(X_scaled)

# ── Profile each cluster ──────────────────────────────────────
profile_cols = [
    'average_playtime', 'playtime_per_dollar', 'rating_ratio',
    'total_ratings', 'owners_mid', 'achievements',
    'has_multiplayer', 'is_free', 'price', 'churned'
]

profile = df.groupby('cluster')[profile_cols].mean().round(3)
profile['game_count'] = df.groupby('cluster')['appid'].count()
print("\nCluster profiles:\n", profile.T)

# ── Assign human-readable segment names ──────────────────────
# We'll name them after seeing the profiles — update these if needed
segment_names = {
    0: 'Multiplayer mid-tier',
    1: 'Mega hits',
    2: 'Casual browsers',
    3: 'Achievement hunters',
    4: 'Free-to-play'
}
df['segment'] = df['cluster'].map(segment_names)

# ── Churn rate per segment ────────────────────────────────────
seg_churn = df.groupby('segment')['churned'].mean().sort_values(ascending=False)
print("\nChurn rate by segment:\n", seg_churn.round(3))

plt.figure(figsize=(8, 5))
colors = ['#534AB7', '#1D9E75', '#D85A30', '#D4537E', '#8C62D9']
seg_churn.plot(kind='bar', color=colors, edgecolor='white', width=0.6)
plt.title('Churn rate by player segment')
plt.xlabel('Segment')
plt.ylabel('Churn rate')
plt.xticks(rotation=25, ha='right')
plt.axhline(y=df['churned'].mean(), color='black', linestyle='--',
            linewidth=1, label=f'Overall avg ({df["churned"].mean():.1%})')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/05_churn_by_segment.png', dpi=150)
plt.show()

# ── PCA scatter plot (2D view of clusters) ───────────────────
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
df['pca_x'] = coords[:, 0]
df['pca_y'] = coords[:, 1]

plt.figure(figsize=(10, 7))
for i, (seg, grp) in enumerate(df.groupby('segment')):
    plt.scatter(grp['pca_x'], grp['pca_y'],
                label=seg, alpha=0.4, s=10, color=colors[i])
plt.title('Player segments — PCA projection')
plt.xlabel('PCA component 1')
plt.ylabel('PCA component 2')
plt.legend(markerscale=3)
plt.tight_layout()
plt.savefig('outputs/06_pca_clusters.png', dpi=150)
plt.show()
print("PCA scatter saved.")

# ── Save final dataset ───────────────────────────────────────
df.to_csv('data/cleaned/steam_clustered.csv', index=False)
print("\nClustered data saved.")