# Player Churn Analysis — Steam Games Dataset

A data analytics project segmenting 27,000+ Steam games into player behavior 
clusters and predicting churn risk using K-Means clustering and logistic 
regression. Built to demonstrate business analyst skills relevant to gaming 
companies like Electronic Arts.

---

## Business Problem

In the gaming industry, player churn — when users stop engaging with a game 
shortly after purchase — is one of the most costly challenges publishers face. 
This project answers three questions:

1. What distinct player engagement segments exist across Steam's game catalog?
2. Which segments are at highest risk of churn?
3. What game attributes most strongly predict whether a player will disengage?

---

## Dataset

- **Source:** Steam Store Games (Full Data) by Nik Davis — Kaggle
- **Size:** 27,075 games, 18 raw features
- **Key columns:** average playtime, owners, price, ratings, genres, categories

---

## Methodology

### 1. Data cleaning
- Dropped 15 rows with missing developer/publisher (0.05% of data)
- Parsed owner ranges (e.g. "10M–20M") into numeric midpoints
- Removed top 1% playtime outliers to reduce clustering skew
- Final cleaned dataset: 26,790 games, 23 features

### 2. Feature engineering
| Feature | Description |
|---|---|
| `playtime_per_dollar` | Average playtime divided by price |
| `rating_ratio` | Positive ratings / total ratings |
| `has_multiplayer` | Binary flag from categories column |
| `is_free` | Binary flag for free-to-play games |
| `churned` | 1 if playtime is below genre's 25th percentile |

### 3. Churn definition
A game is labelled **churned** if its average playtime falls below the 
25th percentile within its primary genre. This genre-relative definition 
accounts for the fact that 2 hours in an RPG signals very different 
engagement than 2 hours in a casual puzzle game.

**Overall churn rate: 77.2%**

### 4. K-Means clustering
- Tested K=2 through K=9 using Elbow Method and Silhouette Score
- Selected K=5 (Silhouette: 0.45) for best business interpretability
- Applied StandardScaler normalization before clustering
- Visualized clusters using PCA 2D projection

### 5. Churn prediction model
- Algorithm: Logistic Regression (scikit-learn)
- Train/test split: 80/20, stratified
- Features: 9 engineered behavioral and product attributes

---

## Key Findings

### Player segments

| Segment | Games | Avg Playtime (hrs) | Churn Rate | Avg Price |
|---|---|---|---|---|
| Mega hits | 516 | 1,033 | 0% | $12.94 |
| Multiplayer mid-tier | 4,117 | 63.5 | 75% | $9.64 |
| Free-to-play | 2,371 | 24.3 | 73% | $0.00 |
| Casual browsers | 19,649 | 44.1 | 81.2% | $5.78 |
| Achievement hunters | 137 | 18.8 | 84.7% | $4.37 |

### Churn model performance

| Metric | Score |
|---|---|
| Accuracy | 97.3% |
| Precision | 96.9% |
| Recall | 99.8% |
| F1 Score | 98.3% |

### Top churn drivers
Features with the strongest negative coefficients (i.e. most protective 
against churn):

1. **Average playtime** — strongest single predictor (coef: -19.9)
2. **Playtime per dollar** — value-for-money drives retention (coef: -13.8)
3. **Owner count** — popular games retain better (coef: -4.9)
4. **Total ratings** — community engagement reduces churn (coef: -2.1)
5. **Price** — only feature that *increases* churn risk (coef: +0.09)

### Genre churn rates (top 5 highest)
- Sports: 87.7%
- Violent: 84.7%
- Casual: 84.2%
- Animation & Modeling: 83.3%
- Indie: 81.2%

---

## Business Recommendations

Based on the analysis, three actionable recommendations for a publisher like EA:

**1. Invest in playtime-driving features early**
Playtime in the first week is the strongest predictor of long-term retention.
Games should prioritize onboarding loops that drive that first 5–10 hours of 
engagement before players disengage.

**2. Re-evaluate the achievement hunter segment**
137 games show extremely high achievement counts but 84.7% churn — suggesting 
achievement systems alone do not drive retention. Publishers should pair 
achievement design with narrative or multiplayer hooks.

**3. Price is a churn risk, not a safety net**
Higher-priced games churn at a slightly higher rate. This suggests players hold 
premium-priced games to a higher standard. Quality and content depth matter 
more than price positioning for retention.

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/satvik/player-churn-analysis.git
cd player-churn-analysis

# Set up environment
python -m venv venv
source venv/bin/activate
pip install pandas numpy matplotlib seaborn scikit-learn plotly

# Download dataset from Kaggle and place in data/raw/steam.csv
# Then run notebooks in order:
python notebooks/01_eda.py
python notebooks/02_cleaning.py
python notebooks/03_features.py
python notebooks/04_clustering.py
python notebooks/05_churn_model.py
```

## Tech Stack
- **Python** — pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Plotly
- **Machine Learning** — K-Means clustering, Logistic Regression, PCA
- **Visualization** — Power BI dashboard (see `/outputs`)
- **Tools** — Cursor, Git, GitHub

