# NLP Airline Sentiment Classifier

> **Multi-class tweet sentiment analysis using a PyTorch ANN with TF-IDF vectorization, advanced text preprocessing, and a two-stage hyperparameter optimization pipeline.**

## Project Overview

This project builds an end-to-end NLP pipeline to classify Twitter airline sentiment into three categories — **Negative**, **Neutral**, and **Positive** — using the [Twitter US Airline Sentiment dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) (~14,640 tweets across 6 US carriers).

The core model is a multi-layer feed-forward Artificial Neural Network (ANN) trained on TF-IDF features, enhanced with class imbalance handling, early stopping, learning rate scheduling, and a systematic two-stage hyperparameter search.

## Repository Structure

```
├── data/
│   └── Tweets.csv
├── nlp_airline_sentiment_classifier.ipynb
├── training_validation_losses.pdf
├── tuned_training_validation_losses.pdf
└── README.md
```

##  Methodology

### 1. Data Preprocessing & EDA
- Removed duplicate rows and irrelevant/high-missingness columns
- Exploratory analysis of sentiment distribution, airline-level breakdowns, negative reason categories, retweet counts, and confidence scores
- Identified and visualized class imbalance (~60% negative, ~20% neutral, ~20% positive)
- Applied **LDA topic modeling** to uncover latent themes in tweet content

### 2. Text Cleaning Pipeline
- HTML stripping (`BeautifulSoup`)
- Emoji conversion to text tokens (`emoji`)
- Regex-based noise removal (mentions, URLs, punctuation)
- Tokenization, stopword removal, and **lemmatization** (`spaCy`, `NLTK`)

### 3. Vectorization
| Method | Library | Notes |
|--------|---------|-------|
| TF-IDF | `scikit-learn` | Primary; sparse → dense |
| Word2Vec | `Gensim` | Mean-pooled sentence vectors |
| FastText | `Gensim` | Mean-pooled; handles OOV |

### 4. ANN Architecture (PyTorch)

```
Input (TF-IDF features)
    │
    ├── Linear(input_dim → hidden_dim1)  + BatchNorm + ReLU + Dropout(p1)
    │
    ├── Linear(hidden_dim1 → hidden_dim2) + BatchNorm + ReLU + Dropout(p2)
    │
    └── Linear(hidden_dim2 → 3)  [Negative | Neutral | Positive]
```

**Key training decisions:**
- **Loss**: `CrossEntropyLoss` with balanced class weights to counteract the 3:1:1 class imbalance
- **Optimizer**: Adam with L2 weight decay
- **Scheduler**: `ReduceLROnPlateau` (factor=0.5, patience=3)
- **Early stopping**: Custom implementation with configurable patience and min-delta
- **Checkpointing**: Best model saved automatically by validation loss

### 5. Hyperparameter Optimization

A two-stage strategy was used to balance exploration and precision:

**Stage 1 — Random Search (50 trials)**

Explored a wide parameter space:

| Hyperparameter | Search Space |
|---|---|
| Learning rate | {1e-2, 5e-3, 1e-3, 5e-4, 1e-4} |
| Weight decay | {1e-2, 1e-3, 1e-4, 1e-5} |
| Hidden dim 1 | {16, 32, 64, 128, 256, 512} |
| Hidden dim 2 | {16, 32, 64, 128, 256} |
| Dropout p1/p2 | {0.1, 0.2, 0.3, 0.4, 0.5, 0.6} |
| Batch size | {16, 32, 64, 128, 256} |

**Stage 2 — Grid Search (focused)**

Built a tight neighbor grid around the top-ranked random trial and exhaustively evaluated all combinations, selecting the configuration maximizing validation F1.

## Results

### Training Dynamics
Both the baseline and tuned models show healthy initial convergence. Train and validation losses closely track each other through epoch ~12, after which train loss continues to decrease while validation loss plateaus around **0.72–0.74** — a moderate overfitting pattern consistent with TF-IDF's limited representational power for nuanced sentiment.

### Final Test Performance (Tuned Model)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Negative | 0.89 | 0.78 | **0.83** | 911 |
| Neutral | 0.44 | 0.57 | **0.50** | 292 |
| Positive | 0.53 | 0.60 | **0.56** | 216 |

**Key observations:**
- **Negative** class performs strongest (F1: 0.83) — dominant class with unambiguous, distinctive vocabulary
- **Neutral** class is the hardest (F1: 0.50) — shares lexical overlap with both other classes, challenging for bag-of-words approaches
- **Positive** class suffers from neutral/positive boundary confusion (55 positives misclassified as neutral)

## 🛠️ Tech Stack

| Category | Libraries |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| NLP preprocessing | `NLTK`, `spaCy`, `BeautifulSoup`, `emoji` |
| Vectorization | `scikit-learn` (TF-IDF), `Gensim` (Word2Vec, FastText) |
| Topic modeling | `scikit-learn` (LDA), `pyLDAvis` |
| Deep learning | `PyTorch` |
| Evaluation | `scikit-learn` (classification_report, confusion_matrix) |
| Visualization | `matplotlib`, `seaborn` |

## Key Design Decisions & Lessons Learned

1. **Class imbalance must be addressed explicitly.** The ~60/20/20 split causes a naive model to over-predict negative. Weighted loss and stratified splits were essential.
2. **TF-IDF has a ceiling on sentiment tasks.** The plateau in validation loss (while training loss keeps falling) signals the feature space isn't expressive enough to capture subtle sentiment shifts. Dense embeddings (Word2Vec, FastText, or fine-tuned transformers) would be the natural next step.
3. **Two-stage HPO is efficient.** Random search identifies promising regions cheaply; grid search then refines. Running full grid search from the start would have been computationally prohibitive given the parameter space.
4. **Neutral is the hard class in sentiment analysis** — a consistent finding in the literature and confirmed here. It is effectively the boundary region of the sentiment spectrum.

## Dataset

**Twitter US Airline Sentiment** — Crowdflower (via Kaggle)  
~14,640 tweets · 6 US carriers · 3 sentiment classes  
[View on Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
