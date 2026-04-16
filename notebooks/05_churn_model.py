import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler

# ── Load data ────────────────────────────────────────────────
df = pd.read_csv('data/cleaned/steam_clustered.csv')

# ── Features & target ────────────────────────────────────────
feature_cols = [
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

X = df[feature_cols].fillna(df[feature_cols].median())
y = df['churned']

# ── Train/test split ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}")

# ── Scale ────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Train model ──────────────────────────────────────────────
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_sc, y_train)

# ── Evaluate ─────────────────────────────────────────────────
y_pred = model.predict(X_test_sc)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)

print(f"\n── Model Performance ──────────────────")
print(f"Accuracy:  {acc:.1%}")
print(f"Precision: {prec:.1%}")
print(f"Recall:    {rec:.1%}")
print(f"F1 Score:  {f1:.1%}")
print(f"\nFull report:\n{classification_report(y_test, y_pred)}")

# ── Confusion matrix ─────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Retained', 'Churned'],
            yticklabels=['Retained', 'Churned'])
plt.title('Confusion matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('outputs/07_confusion_matrix.png', dpi=150)
plt.show()

# ── Feature importance (coefficients) ───────────────────────
coefs = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': model.coef_[0]
}).sort_values('coefficient', ascending=True)

print(f"\nFeature coefficients (negative = reduces churn):\n{coefs}")

plt.figure(figsize=(10, 6))
colors = ['#1D9E75' if c < 0 else '#D85A30' for c in coefs['coefficient']]
plt.barh(coefs['feature'], coefs['coefficient'], color=colors, edgecolor='white')
plt.axvline(x=0, color='black', linewidth=0.8)
plt.title('Churn drivers — logistic regression coefficients')
plt.xlabel('Coefficient (positive = increases churn risk)')
plt.tight_layout()
plt.savefig('outputs/08_feature_importance.png', dpi=150)
plt.show()
print("\nFeature importance chart saved.")

# ── Export summary CSV for Power BI ─────────────────────────
segment_summary = df.groupby('segment').agg(
    game_count        = ('appid',            'count'),
    avg_playtime      = ('average_playtime', 'mean'),
    avg_price         = ('price',            'mean'),
    avg_rating        = ('rating_ratio',     'mean'),
    churn_rate        = ('churned',          'mean'),
    pct_multiplayer   = ('has_multiplayer',  'mean'),
    avg_owners        = ('owners_mid',       'mean')
).round(3).reset_index()

segment_summary.to_csv('outputs/segment_summary.csv', index=False)
print("\nSegment summary exported to outputs/segment_summary.csv")
print(segment_summary)