# Explain Video Engagement Patterns on the MixiGaming Channel Using Interpretable Regression Models

A Big Data Mining research project that analyzes viewer engagement patterns on the **MixiGaming YouTube channel** using interpretable regression models and explainable AI techniques.

The study focuses on identifying how selected video-level and temporal features are associated with three engagement outcomes:

- Views
- Likes
- Comments

---

## Overview

This project analyzes **730 publicly available MixiGaming videos published during 2024–2025**, collected using the **YouTube Data API v3**.

The main objective is not to build a highly accurate forecasting system, but to investigate whether a compact and interpretable feature set can explain engagement patterns under a realistic temporal evaluation setting.

The workflow includes:

- Data collection
- Data preprocessing
- Exploratory Data Analysis
- Feature engineering
- Regression modeling
- Time-based train/test splitting
- Model evaluation
- Residual analysis
- Feature importance
- SHAP-based explainability

---

## Research Question

> Which video-related and temporal factors are associated with viewer engagement on the MixiGaming YouTube channel, and to what extent can historical data from 2024–2025 provide insight into later video performance?

---

## Dataset

The dataset contains **730 videos** from the MixiGaming YouTube channel.

### Main targets

- `view_count`
- `like_count`
- `comment_count`

### Engineered features

- Video duration
- Previous video view count
- Days since previous upload
- Upload hour group
- Title type
- Month sine encoding
- Month cosine encoding

To reduce temporal leakage, sequential features were constructed only from information available before or at the current video.

---

## Data Preprocessing

The preprocessing pipeline includes:

- Duplicate and missing-value checks
- Invalid duration validation
- One-hot encoding for categorical variables
- Cyclical encoding for month
- Min-Max scaling for continuous features
- `log(x + 1)` transformation for engagement targets

The final dataset was divided using a **time-based 80/20 split**:

| Split | Number of Videos |
|---|---:|
| Training | 584 |
| Test | 146 |
| Total | 730 |

Unlike a random split, this setup trains on earlier videos and evaluates on later videos to better reflect temporal generalization.

---

## Models

Two interpretable regression approaches were evaluated:

### Decision Tree Regressor

The primary model was a shallow Decision Tree selected for its ability to capture non-linear relationships while remaining interpretable.

Final configuration:

```python
DecisionTreeRegressor(
    max_depth=3,
    min_samples_split=10,
    min_samples_leaf=10,
    random_state=42
)
