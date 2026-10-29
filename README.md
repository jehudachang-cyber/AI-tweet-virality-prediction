# Predicting Viral AI Tweets: A Comparative Machine Learning Study with a Resampled Stacking Ensemble

## 📌 Project Overview
Most social media posts about Artificial Intelligence (AI) receive minimal interaction, while a small minority go viral. Predicting which posts will capture public attention based solely on content—without relying on author follower counts—is a challenging imbalanced classification problem.

This project analyzes **893,076 AI-related tweets** (820,951 cleaned English tweets from 2017–2021) from the Harvard Dataverse corpus (`doi:10.7910/DVN/NHLEJL`), framing virality prediction as an imbalanced binary classification task targeting the top **11.1%** of posts by combined engagement (likes + retweets + replies)[cite: 37]. 

We engineer a leakage-free machine learning pipeline combining **TF-IDF text vectorization**, **VADER sentiment analysis**, **Random Forest feature selection**, and **SMOTE-Tomek resampling** to benchmark 10 tuned classical algorithms against a **Proposed Resampled Stacking Ensemble (VSE)**[cite: 37]. Findings and model diagnostics are visualized in an interactive **Power BI Dashboard**[cite: 37].
<img width="1121" height="631" alt="POWER BI DASHBOARD" src="https://github.com/user-attachments/assets/da3c0044-42be-491c-a802-75612cbdad44" />

---

## 🔍 Key Analytical & Business Insights
* **Media Multiplier Effect:** Native video/GIF attachments increase the viral rate nearly **6x** over text-only tweets (**43.7%** vs. **7.5%**), while photo attachments reach virality **17.5%** of the time[cite: 37].
* **Sentiment Dynamics:** While overall AI discourse is predominantly positive (71.0%–75.7% across 2017–2021), **neutral/informative tweets go viral most frequently (14.6%)**, outperforming negative (11.9%) and positive (10.2%) posts[cite: 37].
* **Structural & Temporal Drivers:** Random Forest Gini importance reveals that structural and timing attributes (`text_len`, `mention_count`, `word_count`, `hour`, and `hashtag_count`) carry more predictive weight than any single keyword[cite: 37].
* **The Imbalance Mirage:** Standard tree models achieve ~89% accuracy simply by predicting the majority non-viral class[cite: 37]. Our 5-stage ablation study proves that **SMOTE-Tomek resampling** delivers the single largest performance jump (+17.1 F1 points), enabling models to genuinely detect rare viral events[cite: 37].

---

## 🏗️ System Architecture & Leakage-Free Pipeline
1. **Data Acquisition & Cleaning:** Filtered 893,076 raw tweets to 820,951 deduplicated English tweets; stripped URLs/mentions for text processing while preserving expressive punctuation counts[cite: 37].
2. **Feature Engineering (4,014 Features):**
   * **Text Representation:** Unigram and bigram TF-IDF vectors (4,000 terms)[cite: 37].
   * **Lexicon Sentiment:** VADER compound sentiment score (`-1` to `+1`)[cite: 37].
   * **13 Structured Features:** `text_len`, `word_count`, `hashtag_count`, `mention_count`, `has_url`, `has_photo`, `has_video`, `exclam_count`, `question_mark`, `hour`, `dayofweek`, `is_weekend`[cite: 37].
3. **Imbalance-Aware Training Path (80/20 Stratified Split on $n = 12,000$):**
   * **Z-Score Standardization:** Fitted strictly on training folds[cite: 37].
   * **Feature Selection:** Random Forest Gini importance ranking to retain the top 300 features[cite: 37].
   * **Hybrid Resampling:** **SMOTE-Tomek** oversamples the minority class and cleans Tomek-link boundary overlaps (producing 17,014 balanced training instances) strictly inside training folds to prevent data leakage[cite: 37].
4. **Proposed Resampled Stacking Ensemble (VSE):**
   * **Base Learners:** Tuned Logistic Regression, Random Forest, and XGBoost[cite: 37].
   * **Meta-Learner:** Logistic Regression trained on out-of-fold probability predictions via internal 3-fold cross-validation[cite: 37].

---

## 📊 Model Benchmarking & Ablation Results

### Hold-Out Test Performance (11 Models × 9 Metrics)
Models were tuned via leakage-free 3-fold GridSearchCV and evaluated on an untouched held-out test set[cite: 37]:

| Model | Accuracy | Precision | Recall | **F1** | **Macro-F1** | ROC-AUC | **PR-AUC** | **MCC** | **Kappa** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.739 | 0.233 | **0.590** | 0.334 | 0.586 | 0.723 | 0.281 | 0.243 | 0.208 |
| Naive Bayes | 0.802 | 0.248 | 0.387 | 0.302 | 0.593 | 0.651 | 0.194 | 0.199 | 0.193 |
| k-NN | 0.706 | 0.199 | 0.545 | 0.291 | 0.553 | 0.684 | 0.221 | 0.185 | 0.154 |
| Decision Tree | 0.862 | 0.316 | 0.211 | 0.253 | 0.588 | 0.663 | 0.209 | 0.185 | 0.180 |
| Random Forest | 0.885 | 0.464 | 0.244 | 0.320 | 0.629 | 0.742 | 0.344 | 0.280 | 0.264 |
| Extra Trees | 0.887 | 0.478 | 0.241 | 0.320 | 0.629 | **0.748** | 0.344 | 0.284 | 0.266 |
| AdaBoost | 0.837 | 0.311 | 0.391 | 0.347 | 0.627 | 0.715 | 0.301 | 0.257 | 0.255 |
| SVM (RBF) | 0.784 | 0.228 | 0.399 | 0.290 | 0.582 | 0.708 | 0.219 | 0.183 | 0.174 |
| XGBoost | **0.892** | **0.535** | 0.203 | 0.294 | 0.618 | 0.741 | 0.346 | 0.283 | 0.248 |
| MLP (Neural Net) | 0.848 | 0.260 | 0.203 | 0.228 | 0.572 | 0.615 | 0.203 | 0.146 | 0.145 |
| **Proposed VSE (Ours)** | 0.877 | 0.423 | 0.301 | **0.352** | **0.642** | 0.744 | **0.356** | **0.291** | **0.286** |

* **Statistical Significance (10-Fold CV):** Across 10-fold stratified cross-validation, the Proposed VSE achieves **F1 = 0.358 ± 0.057** and **MCC = 0.308 ± 0.057**, outperforming Random Forest ($p < 0.001$, Wilcoxon $p = 0.002$, Cohen's $d = 2.09$) and XGBoost ($p < 0.001$, Wilcoxon $p = 0.002$, Cohen's $d = 2.57$)[cite: 37].

### 5-Stage Incremental Ablation Study
| Stage | Pipeline Configuration | Accuracy | Recall | F1 | MCC | $\Delta$ F1 vs. S0 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **S0** | TF-IDF text only + Random Forest | 0.889 | 0.004 | 0.007 | 0.036 | — |
| **S1** | + 13 Structured & VADER features | 0.890 | 0.008 | 0.015 | 0.082 | +0.75 pp |
| **S2** | + RF Gini Feature Selection (Top 300) | 0.895 | 0.083 | 0.149 | 0.228 | +14.17 pp |
| **S3** | + SMOTE-Tomek Resampling | 0.885 | 0.244 | 0.320 | 0.280 | +31.27 pp |
| **S4** | + **Stacking Ensemble (Full Proposed VSE)** | 0.877 | **0.301** | **0.352** | **0.291** | **+34.42 pp** |

---

## 📈 Interactive Power BI Dashboard
The repository includes a single-page interactive Power BI dashboard (`.pbix`) fed by the pipeline's exported aggregate tables[cite: 37]:
* **Executive KPI Cards:** Total Tweets Analysed (`821K`), Engagement Rate (`46.88%`), Average VADER Sentiment (`0.38`), and Viral Rate (`11.08%`)[cite: 37].
* **Interactive Visuals:** Viral Rate by Sentiment, Yearly Sentiment Mix (2017–2021), Viral Rate by Media Type, Hourly Engagement Rate, and the 11-Model Performance Benchmark (F1 & PR-AUC) with dynamic year slicing[cite: 37].

---

## 🛠️ Tech Stack
* **Language:** Python 3.13[cite: 37]
* **Machine Learning & NLP:** `scikit-learn`, `imbalanced-learn` (SMOTE-Tomek), `XGBoost`, `vaderSentiment`[cite: 37]
* **Data Manipulation & Visualization:** `pandas`, `matplotlib`, **Microsoft Power BI Desktop**[cite: 37]
* **Environment:** Google Colab / Jupyter Notebook (`seed = 42` for full end-to-end reproducibility)[cite: 37]

## 📁 Repository Structure
* `SWA2124_Viral_Analysis.ipynb`: Self-contained Google Colab notebook that downloads the Harvard Dataverse dataset (`doi:10.7910/DVN/NHLEJL`) and reproduces all tables, statistical tests, and figures end-to-end[cite: 37].
* `swa2124_analysis.py`: Data cleaning, feature engineering, descriptive profiling, and Power BI table exports[cite: 37].
* `swa2124_models.py`: Leakage-free resampling pipeline, 3-fold grid search tuning, 10-fold CV, statistical significance testing, ablation study, and VSE stacking implementation[cite: 37].
* `dashboard.pbix`: Interactive Power BI dashboard file[cite: 37].
