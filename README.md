

<div align="center">

# 🚨 Disaster Tweet Classification

### End-to-End NLP Pipeline for Binary Tweet Classification
**Disaster vs. Non-Disaster Detection using Classical Machine Learning**

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-guykaptue-181717?logo=github)](https://github.com/guykaptue)

**Click to open in Colab (notebook) or to open the Academic Report (PDF)t**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18pfgtM_4gbYDSLi68CeHcYzqc0vozaN9?usp=sharing)
[![Academic Report](https://img.shields.io/badge/📄_Academic_Report-PDF-blue?logo=adobeacrobatreader&logoColor=white&labelColor=555)](reports/docs/disaster_tweet_report.pdf)
</div>

---

<details open>
<summary><strong>📋 Table of Contents</strong></summary>

1. [Problem Statement](#-problem-statement)
2. [Project Objectives](#-project-objectives)
3. [Methodology](#-methodology)
4. [Project Structure](#-project-structure)
5. [Installation](#-installation)
6. [Usage](#-usage)
7. [Pipeline Walkthrough](#-pipeline-walkthrough)
   - [Phase 1 — EDA](#phase-1--exploratory-data-analysis)
   - [Phase 2 — Preprocessing](#phase-2--text-preprocessing)
   - [Phase 3 — Vectorisation](#phase-3--text-vectorisation)
   - [Phase 4–5 — Modelling & Tuning](#phase-45--cross-validation--hyperparameter-tuning)
8. [Results](#-results)
9. [Key Findings](#-key-findings)
10. [Limitations](#-limitations)
11. [Future Work](#-future-work)
12. [Acknowledgements](#-acknowledgements)

</details>

---

## 🎯 Problem Statement

Social media platforms — Twitter in particular — have become front-line communication channels during natural disasters, mass-casualty incidents, and other crises. Emergency services, NGOs, insurers, and journalists all need to rapidly identify genuine disaster reports from the overwhelming noise of everyday language.

The challenge is deceptively hard. Consider:

> *"On plus side LOOK AT THE SKY LAST NIGHT IT WAS ABLAZE"*

This describes a spectacular sunset, not a fire — yet a naive keyword model would flag it as a disaster. The inverse problem is equally dangerous: an actual wildfire report that uses understated language could be missed entirely.

This project addresses the **[Kaggle NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)** benchmark: given a tweet's text (and optional keyword/location metadata), automatically classify it as describing a **real disaster event** or **not**.

A 5–10% improvement in classification accuracy can translate to rescue operations being initiated **10–30 minutes earlier** — a potentially life-critical difference.

---

## 🏆 Project Objectives

| Priority | Objective | Target |
|----------|-----------|--------|
| **Primary** | Develop a robust binary classifier for disaster tweets | F1-Score ≥ 0.75 on held-out validation set |
| **Secondary** | Ensure full pipeline transparency and reproducibility | Every design decision justified by data |
| **Tertiary** | Derive actionable deployment recommendations | Business-ready production guidance |

---

## 🔬 Methodology

The solution follows a **six-phase, systematic NLP framework** — from raw CSV to a Kaggle-ready submission — using only classical machine learning (no deep learning).

```
RAW TWEETS  (7,613 training samples)
    │
    ▼
PHASE 1 ── Exploratory Data Analysis (EDA)
           Token frequency · Duplicate detection · Class distribution
           URL patterns · Structural feature profiling
    │
    ▼
PHASE 2 ── Text Preprocessing
           Duplicate & conflict resolution · Structural feature extraction (11 features)
           Tokenisation → Stopword filtering → Lemmatisation → Cleaning
    │
    ▼
PHASE 3 ── Vectorisation  (8 strategies)
           Sparse: TF-IDF Unigram / Bigram / Trigram · Count BOW Unigram / Bigram / Trigram
           Dense:  Word2Vec (100d) · FastText (100d)
    │
    ▼
PHASE 4 ── Cross-Validation Study
           7 classifiers × 8 vectorisers = up to 56 combinations
           Stratified 5-Fold CV · Metrics: Accuracy, Precision, Recall, F1, AUC
    │
    ▼
PHASE 5 ── Hyperparameter Optimisation
           Best-per-vectoriser selection · RandomizedSearchCV (n_iter=30, 5-fold)
    │
    ▼
PHASE 6 ── Final Model Selection & Submission
           Validation-set F1 ranking · Kaggle submission generation
```

<details>
<summary><strong>Model Candidates (click to expand)</strong></summary>

| Model | Library | Configuration |
|-------|---------|---------------|
| Logistic Regression | scikit-learn | `class_weight='balanced'` |
| Naive Bayes (Multinomial) | scikit-learn | Sparse vectors only |
| Linear SVM | scikit-learn | `LinearSVC` |
| Random Forest | scikit-learn | 100 estimators |
| MLP (Neural Network) | scikit-learn | 2-layer feedforward |
| XGBoost | xgboost | `scale_pos_weight=1.35` |
| K-Nearest Neighbours | scikit-learn | — |

</details>

<details>
<summary><strong>Vectorisation Strategies (click to expand)</strong></summary>

| Strategy | Vocabulary | Sparsity | Dimension |
|----------|-----------|---------|----------|
| TF-IDF Unigram | 5,705 | 99.8% | 5,705 |
| TF-IDF Bigram | ~12,000 | 99.9% | ~12,000 |
| TF-IDF Trigram | 18,130 | 99.9% | 18,130 |
| Count BOW Unigram | 5,705 | 99.8% | 5,705 |
| Count BOW Bigram | ~12,000 | 99.9% | ~12,000 |
| Count BOW Trigram | 18,130 | 99.9% | 18,130 |
| Word2Vec (mean pooling) | 5,823 | 0% | 100 |
| FastText (mean pooling) | 5,823 | 0% | 100 |

</details>

---

## 📁 Project Structure

```
disaster_tweets_nlp_classification_project/
│
├── data/
│   ├── raw/
│   │   └── train.csv                          # Original Kaggle training data
│   └── processed/
│       ├── train_clean.csv                    # Deduplicated training set
│       ├── train_preprocessed.csv             # Fully normalised (26 columns)
│       └── test_clean.csv                     # Cleaned test set
│
├── notebook/
│   └── katastrophen_Tweets.ipynb              # Complete reproducible pipeline
│
├── reports/
│   ├── docs/
│   │   └── disaster_tweet_report.pdf          # Full academic report (LaTeX)
│   ├── models/
│   │   └── vectorizer/
│   │       ├── doc2vec/                       # Trained Doc2Vec models
│   │       └── vocabulary/                    # Saved vocabularies (JSON)
│   │           ├── tfidf/
│   │           ├── word2vec/
│   │           ├── doc2vec/
│   │           └── embedding_layer/
│   └── results/
│       ├── confusion_matrix.png
│       ├── confusion_matrix_with_metrics.png
│       ├── holdout_klassenverteilung.png
│       ├── holdout_wahrscheinlichkeitsverteilung.png
│       ├── submission.csv                     # Kaggle submission file
│       ├── cv/                                # CV results (CSV / JSON / PKL)
│       └── plots/
│           ├── models/                        # CV heatmaps, ranking, ROC curves
│           ├── processor/                     # EDA visualisations, word clouds
│           └── vectorizer/                    # PCA, t-SNE, feature weight plots
│
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10 or higher
- Google Colab *(recommended)* or a local Jupyter environment
- Kaggle API credentials for dataset download

### Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn plotly
pip install nltk wordcloud gensim
pip install scikit-learn xgboost
pip install kaggle
```

### Download NLTK Resources

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

### Download Dataset via Kaggle API

```bash
# Ensure kaggle.json is placed at ~/.kaggle/kaggle.json
kaggle competitions download -c nlp-getting-started
unzip nlp-getting-started.zip -d data/raw/
```

---

## 🚀 Usage

Open `notebook/katastrophen_Tweets.ipynb` in **Google Colab** or a local Jupyter environment and execute all cells sequentially. The notebook is fully self-contained and annotated.

> **Note:** Update the `BASE_DIR` path at the top of the notebook to point to your project directory (Google Drive or local path).

<details>
<summary><strong>Step-by-Step Runtime Guide (click to expand)</strong></summary>

| Section | Description | Est. Runtime |
|---------|-------------|-------------|
| Section 3 | Package installation & setup | ~3 min |
| Section 4 | Exploratory Data Analysis | ~2 min |
| Section 5 | Text preprocessing | ~3 min |
| Section 6 | Vectorisation (8 methods) | ~10 min |
| Section 7.1 | Cross-validation study (up to 56 combinations) | ~15–30 min |
| Section 7.6 | Hyperparameter tuning | ~20–40 min |
| Section 7.9 | Final prediction & Kaggle submission | ~1 min |

**Total estimated runtime:** 60–90 minutes on Google Colab (CPU).

</details>

<details>
<summary><strong>Quick Inference Example (click to expand)</strong></summary>

```python
import joblib
from preprocessing import TextPreprocessor

# Load artefacts
vectoriser = joblib.load('reports/models/count_bigram_vectoriser.pkl')
model      = joblib.load('reports/models/logreg_count_bigram_tuned.pkl')

# Preprocess & predict
preprocessor = TextPreprocessor()
clean = preprocessor.normalize("Wildfire spreading rapidly near highway 101")['clean_text']

X    = vectoriser.transform([clean])
prob = model.predict_proba(X)[0][1]
label = "🚨 DISASTER" if prob >= 0.5 else "✅ NON-DISASTER"
print(f"Prediction: {label}  (confidence: {prob:.2%})")
```

</details>

---

## 🔭 Pipeline Walkthrough

### Phase 1 — Exploratory Data Analysis

The EDA phase (notebook Section 4) systematically characterises the raw dataset across four dimensions before any modelling decisions are made.

#### 📊 Class Distribution

The training set contains 7,613 labelled tweets with a **moderate class imbalance**: 57.4% Non-Disaster vs. 42.6% Disaster (ratio = 1.35). This is mild enough that aggressive resampling (e.g. SMOTE) is not warranted; `class_weight='balanced'` handles it cleanly.

![Class Distribution](reports/results/plots/processor/class_distribution.png)

> **Notebook insight (§4.5):** Three complementary perspectives — overall distribution, keyword-filtered distribution (e.g. the ambiguous keyword *"fire"*), and text-length-binned distribution — confirm that disaster tweets are structurally distinct: not only more common in longer texts, but also more tightly clustered around specific event vocabulary.

---

#### 📝 Token Frequency Analysis — Raw Text

Raw token extraction reveals structurally distinct linguistic profiles for the two classes before any preprocessing has taken place.

![Word Frequencies (Raw)](reports/results/plots/processor/word_frequencies_raw.png)

![Word Frequencies (Professional)](reports/results/plots/processor/word_frequencies_professional.png)

> **Notebook insight (§4.3.3):** *Disaster tweets use a reporting, location-oriented language dominated by prepositions (in, on, at) and event words (fire, after). Non-disaster tweets are personal and social, characterised by pronouns (I, you, my, me). Disaster language is concise, factual, and event-oriented — leading to a more right-skewed distribution with many rare tokens but a few very frequent ones.*

---

#### 📊 N-Gram Analysis

Unigram, bigram, and trigram frequency plots reveal how multi-word expressions provide additional classification signal.

![N-Grams Unigram](reports/results/plots/processor/ngrams_1_professional.png)

![N-Grams Bigram](reports/results/plots/processor/ngrams_2_professional.png)

![N-Grams Trigram](reports/results/plots/processor/ngrams_3_professional.png)

> **Notebook insight (§6.4):** Bigram compound expressions — *"fire department"*, *"car accident"*, *"suicide bombing"* — are strong disaster signals that single tokens cannot capture. However, moving to trigrams inflates vocabulary by +218% (5,705 → 18,130 terms) with diminishing returns, as confirmed by the cross-validation study.

---

#### ☁️ Word Clouds

Word clouds provide immediate visual intuition about the dominant vocabulary of each class.

![Word Cloud Comparison (Professional)](reports/results/plots/processor/comparison_wordclouds_professional.png)

![Word Cloud Comparison (Cleaned)](reports/results/plots/processor/comparison_wordclouds_cleaned.png)

> **Notebook insight (§7.8.2):** The false-positive word cloud (non-disaster tweets misclassified as disasters) overlaps heavily with the disaster word cloud — confirming that the model's failure mode is triggered by alarming terms like *"fire"*, *"nuclear"*, *"attack"*, and *"burning"* used in metaphorical or figurative contexts.

---

#### 📏 Text Length Analysis

Disaster tweets are systematically longer (avg. 108 characters vs. 96) and more densely punctuated (3.40 vs. 2.86 per tweet), reflecting a more structured, informative reporting style.

![Text Length Analysis](reports/results/plots/processor/text_length_analysis.png)

---

#### 🗂️ Disaster Category Treemap

The treemap visualises the keyword-based distribution of disaster types, showing which event categories dominate the training data.

![Treemap Categories](reports/results/plots/processor/treemap_categories.png)

---

#### 🐦 Twitter Structural Features

A comparison of Twitter-native structural features across classes confirms strong discriminative signal beyond text content.

![Twitter Features Comparison](reports/results/plots/processor/twitter_features_comparison.png)

> **Notebook insight (§5.3.1):** Disaster tweets contain URLs significantly more often (67.2% vs. 41.7%), carry more hashtags (0.498 vs. 0.383 per tweet), and use more upper-case characters — all consistent with the crisis-reporting and event-tagging behaviour of journalists and official accounts. Non-disaster tweets have more @mentions, reflecting their dialogic, social nature.

---

### Phase 2 — Text Preprocessing

The `TextPreprocessor` class (notebook Section 5) implements a **five-stage normalisation pipeline** with full intermediate persistence. Every stage is saved as a dedicated DataFrame column for transparency and reproducibility.

```
raw_text
  → tokens              [word_tokenize · lowercase · URLs/mentions stripped]
  → tokens_no_stopword  [198-word stoplist; prepositions retained]
  → tokens_lemmatized   [WordNetLemmatizer]
  → tokens_cleaned      [min token length > 1 character]
  → clean_text          [final space-joined string]
```

**Example pipeline trace:**
```
ORIGINAL  : "Latest: USA: Huge sinkhole swallows up Brooklyn intersection http://t.co/..."
tokens    : ['latest', 'usa', 'huge', 'sinkhole', 'swallows', 'up', 'brooklyn', ...]
no_stop   : ['latest', 'usa', 'huge', 'sinkhole', 'swallows', 'brooklyn', 'url']
lemmatized: ['latest', 'usa', 'huge', 'sinkhole', 'swallow', 'brooklyn', 'url']
cleaned   : ['latest', 'usa', 'huge', 'sinkhole', 'swallow', 'brooklyn', 'url']
clean_text: "latest usa huge sinkhole swallow brooklyn url"
```

Additionally, **11 structural features** are extracted from raw text *before* normalisation and combined with Top-30 keyword one-hot encoding to yield a **42-dimensional feature matrix**.

#### Post-Preprocessing Token Distributions

![Text Cleaned & Normalised](reports/results/plots/processor/text_cleaned_normalize.png)

![Word Frequencies (Cleaned)](reports/results/plots/processor/word_frequencies_cleaned.png)

> **Notebook insight (§5.5.1):** After preprocessing, structural placeholders (*token*, *url*, *mention*, *user*) dominate both classes — confirming that the pipeline has correctly unified URLs, @mentions, and user handles. The semantic difference between classes is preserved and even amplified: disaster tokens are more focused (*fire*, *police*, *news*, *california*) while non-disaster tokens remain broadly varied (*like*, *get*, *new*, *day*).

Post-pipeline vocabulary sizes:
- Disaster class: **7,486** unique tokens — focused and homogeneous
- Non-Disaster class: **9,591** unique tokens — broader and more diverse

---

### Phase 3 — Text Vectorisation

Seven text representation strategies are compared end-to-end (notebook Section 6).

#### 📐 Vocabulary & Sparsity Analysis

![Vocabulary Analysis](reports/results/plots/vectorizer/vocab_analysis.png)

![Sparsity Matrix](reports/results/plots/vectorizer/sparsity_matrix_detailed.png)

> **Notebook insight (§6.10.2):** TF-IDF and Count vectors exhibit 99.8–99.9% sparsity — over 99% of each feature matrix is zero. Dense embeddings (Word2Vec, FastText) eliminate this sparsity entirely with compact 100-dimensional vectors, but at the cost of averaging out the precise token-level signals that discriminate disaster from non-disaster tweets.

---

#### 🔑 TF-IDF Feature Weights

Top TF-IDF terms per class confirm semantically meaningful class boundaries.

![TF-IDF Feature Weights](reports/results/plots/vectorizer/tfidf_feature_weights.png)

![Top TF-IDF Terms (Unigram)](reports/results/plots/vectorizer/top_tfidf_terms_tfidf_unigram.png)

![Class-Specific Features](reports/results/plots/vectorizer/class_specific_features.png)

> **Notebook insight (§6.10.4):** Disaster tweets score highest TF-IDF on specific catastrophe terms (*"fire"*, *"disaster"*, *"wildfire"*), while non-disaster tweets are characterised by social and filler words (*"user"*, *"like"*, *"lol"*). This clear separation is the foundation for why sparse linear models outperform complex non-linear ones on this corpus.

---

#### 📉 TF-IDF: PCA & t-SNE Visualisations

Two-dimensional projections of the TF-IDF feature space. The approximate **linear separability** visible in PCA explains why Logistic Regression outperforms non-linear models.

![TF-IDF PCA](reports/results/plots/vectorizer/tfidf_pca.png)

![TF-IDF t-SNE](reports/results/plots/vectorizer/tfidf_tsne.png)

**All TF-IDF variants compared (Unigram / Bigram / Trigram):**

![Combined PCA t-SNE TF-IDF Variants](reports/results/plots/vectorizer/combined_pca_tsne_tfidf_variants.png)

> **Notebook insight (§6.12.1):** The PCA plot shows a cigar-shaped cluster (generic/short/low-information terms near the origin) flanked by two more spread clusters. The t-SNE reveals non-linear structure not visible in PCA — discrete micro-clusters emerge for specific disaster event types (wildfire, earthquake, flood), but significant overlap remains in the middle of the space.

**All Count BOW variants compared:**

![Combined PCA t-SNE Count Variants](reports/results/plots/vectorizer/combined_pca_tsne_count_variants.png)

---

#### 🧠 Word2Vec Embeddings

Word2Vec (Skip-Gram, 100d) learns dense semantic representations. Words in similar contexts end up near each other in the 100-dimensional space.

![Word2Vec PCA](reports/results/plots/vectorizer/word2vec_pca.png)

![Word2Vec t-SNE](reports/results/plots/vectorizer/word2vec_tsne.png)

**Word-level semantic map** — semantically related words cluster together:

![Word2Vec Word PCA](reports/results/plots/vectorizer/word2vec_word_pca.png)

![Word2Vec Word t-SNE](reports/results/plots/vectorizer/word2vec_word_tsne.png)

**Full 2×2 grid (PCA 2D/3D + t-SNE 2D/3D):**

![2x2 PCA/t-SNE Word2Vec](reports/results/plots/vectorizer/pca_tsne_2x2_word2vec_mean.png)

> **Notebook insight (§6.12.3):** PC1 explains 29.2% of variance, PC2 explains 11.7% — a total of 41.0%. There is considerable overlap between classes in the Word2Vec space, suggesting that mean-pooled document embeddings lose the precise token-level signals that make TF-IDF so effective for this task.

---

#### ⚡ FastText Embeddings

FastText extends Word2Vec with subword character n-grams, giving it 100% vocabulary coverage and robustness to typos (*"eqrthquake"* → *"earthquake"*).

![FastText PCA](reports/results/plots/vectorizer/glove_pca.png)

![FastText t-SNE](reports/results/plots/vectorizer/glove_tsne.png)

**Full 2×2 grid:**

![2x2 PCA/t-SNE FastText](reports/results/plots/vectorizer/pca_tsne_2x2_fasttext_mean.png)

> **Notebook insight (§6.12.4):** FastText explains more variance than Word2Vec (PC1+PC2 = 54.8% vs. 41.0%). It produces higher L2-norms for disaster tweets — suggesting better embedded class separation. However, this advantage does not translate to better downstream classification because mean-pooling still dilutes the sharp token signals that TF-IDF preserves directly.

---

#### 📊 Embedding Distribution: Word2Vec vs. FastText

Value distributions (KDE) and L2-norms of document vectors split by class.

![Embedding Distribution](reports/results/plots/vectorizer/embedding_distribution.png)

> **Notebook insight (§6.11.1):** Both embedding methods show symmetric, bell-shaped distributions — a sign of stable, well-trained vectors. Disaster tweets have significantly higher L2-norms in both methods, especially FastText. This class-conditioned norm difference is the source of FastText's slight advantage over Word2Vec, even though both remain weaker than TF-IDF.

---

#### 📊 Feature Weight Distribution

Distribution of feature weights across all vectorisation methods.

![Feature Weight Distribution](reports/results/plots/vectorizer/feature_weight_distribution_detailed.png)

---

#### 📋 Methods Comparison Dashboard

A unified overview comparing all 8 vectorisation strategies across vocabulary size, sparsity, dimensionality, and class distribution.

![Methods Comparison Dashboard](reports/results/plots/vectorizer/methods_dashboard.png)

---

### Phase 4–5 — Cross-Validation & Hyperparameter Tuning

#### 🏆 Model Ranking Plot

All model–vectoriser combinations ranked by mean 5-fold CV F1-score. The colour gradient encodes performance — darker blue = stronger model.

![CV Model Ranking](reports/results/plots/models/cv_model_ranking_blue.png)

> **Notebook insight (§7.2.1):** LogReg + TF-IDF Unigram leads with **F1 = 0.7589**. Adding bigram/trigram context marginally hurts performance (−0.4% to −1.0%) — the extra N-grams introduce more noise than signal. Naive Bayes with TF-IDF is a solid runner-up (F1 = 0.7505), confirming that the task is well-suited to linear probabilistic models. Complex models (XGBoost, MLP, Random Forest) consistently underperform their linear counterparts on every vectoriser.

---

#### 🗺️ Performance Heatmap: Model × Vectoriser

The heatmap condenses all 49 combinations into one visual — making the linear model dominance on sparse features immediately apparent across both F1 and AUC.

![CV Heatmap Grouped](reports/results/plots/models/cv_heatmap_mean_f1_grouped.png)

![CV Heatmap](reports/results/plots/models/cv_heatmap_mean_f1.png)

> **Notebook insight (§7.2.2):** The pattern is consistent across both heatmaps: top-left quadrant (linear models × sparse features) is uniformly dark; bottom-right quadrant (complex models × dense embeddings) is uniformly light. Neither more complex models nor richer semantic representations offer any advantage on this short-text, lexically-dominated classification task.

---

#### 📉 ROC Curves — All 4 Baseline Models

The ROC curve plots **True Positive Rate (sensitivity)** vs. **False Positive Rate** across all decision thresholds from 0 to 1. The higher and further left a curve hugs, the better.

![Baseline ROC Curves](reports/results/plots/models/baseline_roc_curves.png)

> **Notebook insight (§7.5.3):** *At low thresholds (high sensitivity zone): TF-IDF + LogReg rises fastest → it achieves the highest early TPR while keeping FPR low. This is critical for emergency services who need to catch as many real disasters as possible. FastText + SVM is notably flatter throughout, confirming its weaker discriminative ability.* The best AUC is achieved by TF-IDF + LogReg (0.866), meaning this model maintains the most reliable class separation across all operating points.

**How to read this chart:**
- Each point on a curve = a different classification threshold
- Moving left to right = lowering the threshold (classifying more tweets as disaster)
- The diagonal = random guessing (AUC = 0.5)
- Our best model: AUC = **0.866**

---

#### 📊 Baseline vs. Tuned Performance

Side-by-side F1 and ROC-AUC comparisons before and after `RandomizedSearchCV` tuning (30 iterations, 5-fold):

![Baseline vs Tuned F1](reports/results/plots/models/baseline_vs_tuned_f1.png)

![Baseline vs Tuned ROC-AUC](reports/results/plots/models/baseline_vs_tuned_roc_auc.png)

#### ΔF1 Gain from Tuning

![F1 Gain Tuning](reports/results/plots/models/f1_gain_tuning.png)

> **Notebook insight (§7.7.2):** The largest tuning gain goes to Count Bigram × LogReg (+0.0112 F1), which becomes the final champion. TF-IDF + LogReg shows virtually no change from tuning — its default configuration was already near-optimal, confirming the structural fit between L2 logistic regression and high-dimensional sparse TF-IDF features.

#### 📡 Radar Chart: Baseline vs. Tuned (All Metrics)

![Radar Chart Baseline vs Tuned](reports/results/plots/models/radar_baseline_vs_tuned.png)

> **Notebook insight (§7.7.3):** All four model families occupy a stable performance band (0.6–0.8 across all metrics). The radar chart confirms there is no single-metric outlier — the models are well-rounded rather than trading one metric for another. Tuning produces a small but consistent improvement in the count-based model without degrading any dimension.

---

#### 🔍 Feature Importance

LogReg coefficients reveal exactly which tokens drive disaster vs. non-disaster predictions.

![Feature Importance](reports/results/plots/models/feature_importance_TF-IDF_—_tfidf_unigram_x_LogReg.png)

> **Notebook insight (§7.8.1):** Terms like *"hiroshima"*, *"fire"*, *"wildfire"*, and *"earthquake"* are the strongest disaster predictors. Geographic terms (*"california"*, *"japan"*) also rank highly — the model has learned that certain regions have higher disaster frequency. On the non-disaster side, emotional and social language (*"lol"*, *"cute"*, *"beautiful"*) dominates the negative coefficients.

---

#### ❌ Error Analysis: False Positives & False Negatives

Word clouds of misclassified tweets expose the model's systematic failure modes.

![Error Word Clouds](reports/results/plots/models/error_wordclouds.png)

> **Notebook insight (§7.8.2):** **False Positives** (non-disaster tweets misclassified as disasters): the model is triggered by alarming terms (*"fire"*, *"nuclear"*, *"attack"*, *"emergency"*, *"burning"*) regardless of their context — a direct consequence of the bag-of-words assumption. **False Negatives** (real disasters missed): tweets lacking canonical disaster keywords or using indirect, context-dependent language to describe events. Both failure modes are inherent to lexical models and require transformer-based architectures (BERT, RoBERTa) for meaningful improvement.

---

## 📊 Results

### Final Model Performance

| Metric | Value |
|--------|-------|
| **F1-Score (Validation)** | **0.7715** |
| **ROC-AUC (Validation)** | **0.8625** |
| Accuracy | 0.814 |
| Precision (Disaster class) | 0.81 |
| Recall (Disaster class) | 0.74 |

**Winner:** `Count Bigram × Logistic Regression`
**Hyperparameters:** `solver=saga` · `penalty=l2` · `C=0.5` · `max_iter=2000`

---

### System Comparison

| Rank | System | F1 | AUC |
|------|--------|----|-----|
| 🥇 | Count Bigram × Logistic Regression *(final)* | **0.7715** | 0.8625 |
| 🥈 | TF-IDF Unigram × Logistic Regression | 0.7711 | **0.8661** |
| 🥉 | TF-IDF Unigram × Linear SVM | ~0.765 | ~0.855 |
| 4 | Count BOW × Logistic Regression | ~0.762 | ~0.851 |
| 5 | Word2Vec × Linear SVM | 0.743 | 0.852 |
| 6 | FastText × Linear SVM | 0.715 | 0.820 |

---

### Confusion Matrix

The confusion matrix breaks down predictions on the validation set into four outcome types.

![Confusion Matrix with Metrics](reports/results/confusion_matrix_with_metrics.png)

> **Notebook insight (§7.8.3):** With 81.4% accuracy and ROC-AUC 0.862, the model demonstrates strong discriminative ability. The 74% recall on the Disaster class is the key operational metric for emergency services — meaning 26% of real disaster tweets are missed and require human review.

**How to read this chart:**
| | Predicted: Non-Disaster | Predicted: Disaster |
|---|---|---|
| **Actual: Non-Disaster** | ✅ True Negative | ❌ False Positive (metaphorical language) |
| **Actual: Disaster** | ❌ False Negative (indirect language) | ✅ True Positive |

---

### Holdout Prediction Distribution

The probability histogram for the unlabelled Kaggle test set reveals the model's confidence profile in production conditions.

![Holdout Probability Distribution](reports/results/holdout_wahrscheinlichkeitsverteilung.png)

![Holdout Class Distribution](reports/results/holdout_klassenverteilung.png)

> **Notebook insight (§7.9.4):** Two clear confidence clusters emerge: ~12–13% of tweets at low risk (p < 0.3) and ~14% at high risk (p > 0.8) — where the model is most reliable. The ~36% uncertainty zone (0.3 ≤ p < 0.7) is where errors concentrate and human review adds the most value. This directly motivates the three-tier confidence routing strategy in the deployment recommendations.

---

## 💡 Key Findings

<details open>
<summary><strong>1. Simplicity outperforms complexity in this domain</strong></summary>

Logistic Regression with sparse Count/TF-IDF features **consistently outperforms** XGBoost, MLP, and Random Forest across all 49 experimental combinations. Four structural reasons explain this:

- **Short texts, high signal density.** Every word in a tweet carries proportionally more predictive weight than in long documents. Embedding mean-pooling averages this signal away.
- **Domain-specific vocabulary.** Disaster tweets occupy a clearly bounded lexical field. TF-IDF identifies class-specific terms (`fire`, `earthquake`, `wildfire`) with minimal noise.
- **Linear separability.** PCA plots confirm approximate linear separability in TF-IDF space — non-linear boundaries are unnecessary and introduce overfitting.
- **Keyword dominance.** The most predictive features are lexical, not positional or compositional — exactly the signal type bag-of-words models are built for.

</details>

<details>
<summary><strong>2. Systematic evaluation is non-negotiable</strong></summary>

Without testing all 49 combinations, the intuitive default choice (XGBoost + FastText embeddings) would have been selected — but it performs **5.7% below** the actual best model. Data-driven model selection is essential: the heatmap confirms this pattern holds consistently across all metrics and all cross-validation folds.

</details>

<details>
<summary><strong>3. Interpretability and performance are not in conflict</strong></summary>

The best model is also the most explainable. Logistic Regression coefficients directly quantify each feature's contribution:

- **→ Disaster:** `suicide bombing`, `wildfire`, `earthquake`, `hiroshima`, `california`, `derailment`
- **→ Non-Disaster:** `lol`, `cute`, `beautiful`, `love`, `video`

This interpretability is critical for stakeholders operating in safety-critical or regulatory contexts who must justify automated decisions.

</details>

<details>
<summary><strong>4. The confidence distribution unlocks production strategy</strong></summary>

The bimodal probability distribution (confident predictions at both extremes, ~36% uncertain middle) directly motivates a three-tier routing architecture:

| Confidence Zone | p range | Action | Share |
|-----------------|---------|--------|-------|
| High confidence | p ≥ 0.75 | Automated processing | ~26% |
| Uncertain | 0.40 ≤ p < 0.75 | Human review queue | ~36% |
| Low risk | p < 0.40 | Discard / no action | ~38% |

This reduces human workload by ~74% while maintaining oversight where the model is least reliable.

</details>

---

## ⚠️ Limitations

<details>
<summary><strong>Data Limitations (click to expand)</strong></summary>

- **Temporal bias** — The dataset captures a specific historical period. Social media language evolves rapidly; performance may degrade without periodic retraining.
- **Label noise** — 110 duplicate tweets with conflicting labels were found and resolved; further annotation noise likely remains (inter-annotator agreement unknown).
- **English only** — Disaster events in other languages are not covered by this pipeline.
- **Selection bias** — The Kaggle dataset may over-represent certain disaster types or geographic regions.

</details>

<details>
<summary><strong>Modelling Limitations (click to expand)</strong></summary>

- **Metaphorical language** — Expressions like *"the market is on fire"* or *"this performance was explosive"* systematically confuse bag-of-words models (visible in the false-positive word cloud).
- **Irony and sarcasm** — *"I am literally dying from cringe"* cannot be resolved without deep contextual understanding; requires transformer-based architectures.
- **No temporal or geospatial features** — Tweet timestamps, trending patterns, and geolocation are not utilised despite likely carrying strong disaster signal.
- **~23% error rate** — At F1 = 0.77, approximately one in four tweets is misclassified, requiring human review workflows in production.

</details>

---

## 🔮 Future Work

| Priority | Direction | Expected Benefit |
|----------|-----------|-----------------|
| 🔴 High | Fine-tune `bertweet-base` or `roberta-base` (Twitter-pretrained) | F1 +5–10% (est. ~0.83) |
| 🔴 High | Confidence-based 3-tier routing (auto / review / escalate) | Reduced false-alarm rate in production |
| 🟡 Medium | Threshold optimisation per stakeholder use-case | Tunable Recall/Precision trade-off |
| 🟡 Medium | Multimodal features: geolocation, timestamp, retweet rate | +2–4% F1 from non-text signals |
| 🟡 Medium | Continuous retraining pipeline on live data | Sustained performance over time |
| 🟢 Low | Multi-class extension: disaster type + severity level | More actionable resource allocation |
| 🟢 Low | Multilingual extension | Global disaster event coverage |
| 🟢 Low | Real-time streaming integration (Kafka / Kinesis) | Production deployment at scale |

---

## 🙏 Acknowledgements

- **Author:** Guy M. Kaptue T. — [github.com/guykaptue](https://github.com/guykaptue)
- **Dataset:** [Kaggle — Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)
- **Core Libraries:** [scikit-learn](https://scikit-learn.org), [NLTK](https://www.nltk.org), [Gensim](https://radimrehurek.com/gensim/), [XGBoost](https://xgboost.readthedocs.io/)
- **Visualisation:** [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), [Plotly](https://plotly.com/)

---

<div align="center">

**© 2026 Guy M. Kaptue T. — All Rights Reserved**
*Built with ❤️ and Python for the Kaggle NLP with Disaster Tweets Competition*

*For a full academic treatment — including statistical analysis, all visualisations, and formal methodology documentation — see [`reports/docs/disaster_tweet_report.pdf`](reports/docs/disaster_tweet_report.pdf).*

</div>