import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load data ──────────────────────────────────────────────
df = pd.read_csv('data/raw/steam.csv')

# ── Basic inspection ───────────────────────────────────────
print("Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nFirst 5 rows:\n", df.head())

# ── Missing values ─────────────────────────────────────────
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({
    'missing_count': missing,
    'missing_pct': missing_pct
}).query('missing_count > 0').sort_values('missing_pct', ascending=False)

print("\nMissing values:\n", missing_report)

# ── Summary stats ──────────────────────────────────────────
print("\nSummary stats:\n", df.describe())

# ── Quick distribution plots ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Playtime distribution (log scale because it's heavily skewed)
axes[0].hist(np.log1p(df['average_playtime']), bins=50, color='#534AB7', edgecolor='white')
axes[0].set_title('Avg playtime (log scale)')
axes[0].set_xlabel('log(playtime + 1)')

# Price distribution
axes[1].hist(df['price'].dropna(), bins=50, color='#1D9E75', edgecolor='white')
axes[1].set_title('Price distribution')
axes[1].set_xlabel('Price (USD)')

# Rating (positive ratio)
if 'positive_ratings' in df.columns and 'negative_ratings' in df.columns:
    df['rating_ratio'] = df['positive_ratings'] / (df['positive_ratings'] + df['negative_ratings'] + 1)
    axes[2].hist(df['rating_ratio'].dropna(), bins=50, color='#D85A30', edgecolor='white')
    axes[2].set_title('Positive rating ratio')
    axes[2].set_xlabel('Ratio')

plt.tight_layout()
plt.savefig('outputs/01_distributions.png', dpi=150)
plt.show()
print("\nPlot saved to outputs/01_distributions.png")