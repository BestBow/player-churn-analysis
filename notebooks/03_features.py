import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load cleaned data ────────────────────────────────────────
df = pd.read_csv('data/cleaned/steam_clean.csv')

# ── Select features for clustering ──────────────────────────
# These are the columns that describe player engagement behavior
cluster_features = [
    'average_playtime',
    'playtime_per_dollar',
    'rating_ratio',
    'total_ratings',
    'owners_mid',
    'achievements',
    'has_multiplayer',
    'is_free',
    'price'
]

X = df[cluster_features].copy()

# ── Handle any remaining nulls in feature set ────────────────
X = X.fillna(X.median())
print("Feature matrix shape:", X.shape)
print("\nFeature stats:\n", X.describe().round(2))

# ── Scale features ───────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=cluster_features)

print("\nScaled feature sample:\n", X_scaled_df.head())

# ── Correlation heatmap ──────────────────────────────────────
plt.figure(figsize=(10, 8))
corr = X.corr().round(2)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, square=True)
plt.title('Feature correlation matrix', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/02_correlation_heatmap.png', dpi=150)
plt.show()
print("\nHeatmap saved.")

# ── Churn rate by genre (top 15 genres) ─────────────────────
genre_churn = (
    df.groupby('primary_genre')['churned']
    .agg(['mean', 'count'])
    .rename(columns={'mean': 'churn_rate', 'count': 'game_count'})
    .query('game_count >= 50')
    .sort_values('churn_rate', ascending=False)
    .head(15)
)

print("\nChurn rate by genre (min 50 games):\n", genre_churn.round(3))

plt.figure(figsize=(12, 6))
bars = plt.barh(genre_churn.index, genre_churn['churn_rate'],
                color='#534AB7', edgecolor='white')
plt.axvline(x=df['churned'].mean(), color='#D85A30', linestyle='--',
            linewidth=1.5, label=f'Overall avg ({df["churned"].mean():.1%})')
plt.xlabel('Churn rate')
plt.title('Churn rate by genre (genres with 50+ games)')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/03_churn_by_genre.png', dpi=150)
plt.show()
print("Genre churn chart saved.")

# ── Save scaled features for clustering ─────────────────────
X_scaled_df.to_csv('data/cleaned/features_scaled.csv', index=False)
df.to_csv('data/cleaned/steam_clean.csv', index=False)
print("\nScaled features saved to data/cleaned/features_scaled.csv")