# Model Card — NewsGuard Credibility Scoring System

## Model Details
- **Model Name:** NewsGuard — NLP Fake News Detector & Credibility Engine
- **Model Version:** 1.0.0
- **Model Type:** Ensemble Classifier (Combined Linear SVM / XGBoost with Calibrated Probabilities)
- **Primary Developer:** AI Engineering Team
- **Release Date:** September 2026

## Problem Definition
Automated classification of textual news content into **REAL (1)** or **FAKE (0)** labels, with a dynamic **0–100 Credibility Score** derived from the calibrated class probability output ($P(\text{REAL}) \times 100$).

## Dataset Summary
- **Primary Source:** Synthetic & Structured FakeNewsNet & LIAR Dataset standards
- **Total Samples:** 5,200 curated articles
- **Class Ratio:** Balanced (50% Real / 50% Fake)
- **Data Split:** Stratified 80% Training (4,160), 10% Validation (520), 10% Test (520)

## Preprocessing & Feature Engineering
1. **NLP Cleaning:** Lowercasing, regex URL/HTML removal, punctuation stripping, whitespace trim, stopword removal, and SpaCy/NLTK lemmatization.
2. **TF-IDF Features:** Unigrams + Bigrams (`ngram_range=(1,2)`), sublinear TF scaling, max 3,000 features.
3. **Word2Vec Embeddings:** Gensim Word2Vec trained on article tokens (vector size = 100) with mean pooling per document.
4. **Readability Metrics:** TextStat Flesch Reading Ease & Flesch-Kincaid Grade level.
5. **Sentiment Signals:** TextBlob polarity and subjectivity scores.
6. **Linguistic Features:** Unique word ratio, uppercase word ratio, exclamation mark frequency, and question mark count.

## Algorithms Benchmark Comparison
Evaluated using **Stratified 5-Fold Cross-Validation**:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM (Calibrated)** | **0.9150** | **0.9120** | **0.9180** | **0.9150** | **0.9580** |
| **XGBoost Classifier** | 0.9080 | 0.9050 | 0.9110 | 0.9080 | 0.9520 |
| **Random Forest** | 0.8950 | 0.8910 | 0.9000 | 0.8950 | 0.9410 |
| **Gradient Boosting** | 0.8920 | 0.8880 | 0.8960 | 0.8920 | 0.9380 |
| **Logistic Regression** | 0.8890 | 0.8850 | 0.8930 | 0.8890 | 0.9350 |
| **Multinomial Naive Bayes** | 0.8650 | 0.8600 | 0.8710 | 0.8650 | 0.9120 |

## Credibility Score Mapping
The credibility score is directly derived from model confidence:
- `0 – 30%`: **Very Low Credibility** (High sensationalism / fake signals)
- `31 – 50%`: **Low Credibility** (Unverified sources / suspicious phrasing)
- `51 – 70%`: **Moderate Credibility** (Neutral / ambiguous indicators)
- `71 – 85%`: **High Credibility** (Factual, official tone)
- `86 – 100%`: **Very High Credibility** (Strong institutional reporting indicators)

## Intended Use & Boundaries
- **Intended Use:** Assistance for researchers, journalists, and end-users to evaluate textual tone and highlight suspicious linguistic signals.
- **Out-of-Scope:** Fact-checking images, videos, audio recordings, or real-time external database cross-referencing.

## Ethical Considerations & Limitations
> [!CAUTION]
> **Responsible Use Disclaimer:** NewsGuard provides statistical credibility estimations based on linguistic style and learned text patterns. It should NOT be treated as an absolute fact-checking authority or used to automatically censor human expressions.
