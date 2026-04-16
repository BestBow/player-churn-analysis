import pandas as pd
import numpy as np

# ── Load raw data ───────────────────────────────────────────
df = pd.read_csv('data/raw/steam.csv')

# ── Drop rows with missing developer/publisher ──────────────
df = df.dropna(subset=['developer', 'publisher'])
print(f"After dropping missing: {df.shape}")

# ── Parse owners range → numeric midpoint ───────────────────
def parse_owners(owners_str):
    try:
        low, high = owners_str.replace(',', '').split('-')
        return (int(low) + int(high)) / 2
    except:
        return np.nan

df['owners_mid'] = df['owners'].apply(parse_owners)

# ── Parse release_date → year ───────────────────────────────
df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year

# ── Extract primary genre (first one listed) ────────────────
df['primary_genre'] = df['genres'].str.split(';').str[0].str.strip()

# ── Extract platform flags ───────────────────────────────────
df['has_windows'] = df['platforms'].str.contains('windows', case=False).astype(int)
df['has_mac']     = df['platforms'].str.contains('mac',     case=False).astype(int)
df['has_linux']   = df['platforms'].str.contains('linux',   case=False).astype(int)

# ── Multiplayer flag from categories ────────────────────────
df['has_multiplayer'] = df['categories'].str.contains('Multi-player', case=False, na=False).astype(int)

# ── is_free flag ─────────────────────────────────────────────
df['is_free'] = (df['price'] == 0).astype(int)

# ── rating_ratio ─────────────────────────────────────────────
df['total_ratings'] = df['positive_ratings'] + df['negative_ratings']
df['rating_ratio']  = df['positive_ratings'] / (df['total_ratings'] + 1)

# ── playtime_per_dollar ──────────────────────────────────────
# For free games, use price=1 to avoid division by zero
df['playtime_per_dollar'] = df['average_playtime'] / df['price'].replace(0, 1)

# ── Define CHURN ─────────────────────────────────────────────
# Churn = game's average_playtime is below the 25th percentile
# within its primary genre. Low playtime relative to peers = churned.
genre_p25 = df.groupby('primary_genre')['average_playtime'].transform(lambda x: x.quantile(0.25))
df['churned'] = (df['average_playtime'] <= genre_p25).astype(int)

print(f"\nOverall churn rate: {df['churned'].mean():.1%}")
print(f"Churned games:      {df['churned'].sum():,}")
print(f"Retained games:     {(df['churned'] == 0).sum():,}")

# ── Remove extreme outliers ──────────────────────────────────
# Cap playtime at 99th percentile to avoid skewing clusters
p99 = df['average_playtime'].quantile(0.99)
df = df[df['average_playtime'] <= p99]
print(f"\nAfter removing playtime outliers: {df.shape}")

# ── Drop columns we no longer need ──────────────────────────
cols_to_drop = ['owners', 'platforms', 'categories', 'steamspy_tags', 
                'release_date', 'genres', 'median_playtime']
df = df.drop(columns=cols_to_drop)

# ── Save cleaned data ────────────────────────────────────────
df.to_csv('data/cleaned/steam_clean.csv', index=False)
print(f"\nCleaned data saved. Final shape: {df.shape}")
print(f"\nFinal columns:\n{df.columns.tolist()}")