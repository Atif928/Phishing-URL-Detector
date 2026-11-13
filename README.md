# Phishing URL Detector

**Machine Learning-Based Malicious URL Classification System**

A complete, end-to-end machine learning project that classifies a URL as
**phishing/malicious** or **legitimate/benign** using lexical features extracted
from the URL string — plus a Streamlit web application that serves the trained
model.

> **Educational project.** This is a URL-string classifier, not a complete
> cybersecurity product. It never visits the submitted URL and cannot guarantee
> that a website is safe or malicious.

---

## Table of contents

1. [Introduction](#1-introduction)
2. [Problem statement](#2-problem-statement)
3. [Project objectives](#3-project-objectives)
4. [Features](#4-features)
5. [Results at a glance](#5-results-at-a-glance)
6. [Technology stack](#6-technology-stack)
7. [Dataset](#7-dataset)
8. [Feature engineering](#8-feature-engineering)
9. [Exploratory data analysis](#9-exploratory-data-analysis)
10. [Machine learning algorithms](#10-machine-learning-algorithms)
11. [Evaluation methodology](#11-evaluation-methodology)
12. [Project architecture](#12-project-architecture)
13. [Folder structure](#13-folder-structure)
14. [Installation on Windows](#14-installation-on-windows)
15. [Dataset setup](#15-dataset-setup)
16. [Training and evaluation commands](#16-training-and-evaluation-commands)
17. [Running the application](#17-running-the-application)
18. [Running the tests](#18-running-the-tests)
19. [Screenshots](#19-screenshots)
20. [Limitations](#20-limitations)
21. [Future improvements](#21-future-improvements)
22. [Troubleshooting](#22-troubleshooting)
23. [Dataset attribution and license](#23-dataset-attribution-and-license)
24. [Disclaimer](#24-disclaimer)

---

## 1. Introduction

Phishing attacks trick people into entering passwords or payment details on
websites that imitate trusted brands. The URL is often the first visible clue:
attackers use misleading domains, long hostnames, brand names on unofficial
domains, unusual top-level domains and credential-related keywords.

This project builds a machine learning pipeline that learns those patterns from
a public dataset of real, labelled URLs. Features are computed **only from the
URL text** — never by loading the page — and a Scikit-learn model produces a
phishing/legitimate prediction with a probability score. A Streamlit app exposes
the model through a dashboard, a prediction page and a model-performance page.

## 2. Problem statement

Given a URL string, predict whether it is more likely to be:

- **Phishing / malicious** (class `1`), or
- **Legitimate / benign** (class `0`).

The system must be reproducible, evaluated on unseen data, and deployable as a
web application — while making clear that a prediction is not a safety
guarantee.

## 3. Project objectives

- Collect and document a public dataset that contains **actual URL strings**.
- Clean and validate URLs, normalise labels and remove duplicates.
- Engineer meaningful numeric features from each URL.
- Perform exploratory data analysis and save readable charts.
- Train and compare **Logistic Regression**, **Decision Tree** and
  **Random Forest** on identical data.
- Evaluate with accuracy, precision, recall, F1-score, ROC-AUC, confusion
  matrices and training time.
- Select the best model using appropriate metrics — not accuracy alone.
- Persist the model together with its exact feature schema.
- Serve predictions through a clean Streamlit application.
- Cover the pipeline with automated tests and beginner-friendly documentation.

## 4. Features

| Area | What is included |
|---|---|
| Data | 235,795 real labelled URLs from PhiUSIIL (UCI ML Repository) |
| Cleaning | Missing/duplicate/invalid URL removal, label normalisation, class report |
| Features | 38 lexical URL features (length, characters, host structure, risk signals) |
| EDA | Class balance, length distributions, special characters, correlation heatmap, feature comparison, TLD analysis |
| Models | Logistic Regression, Decision Tree, Random Forest with balanced class weights |
| Evaluation | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, training time, ROC curves, comparison charts |
| Persistence | Joblib model + metadata with feature names, order and schema hash |
| Web app | Dashboard, URL prediction, model performance, about/limitations pages |
| Safety | URL validation, no network requests, no JavaScript, no reputation APIs |
| Quality | 57 pytest tests, friendly error handling, Windows instructions |

## 5. Results at a glance

Measured on the **held-out test set**: 47,073 URLs
(26,970 legitimate, 20,103 phishing) that were not used for training.

| Rank | Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Train time |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | **Random Forest** | **0.9955** | **0.9987** | **0.9908** | **0.9948** | **0.9981** | 50.25 s |
| 2 | Decision Tree | 0.9954 | 0.9985 | 0.9907 | 0.9946 | 0.9969 | 2.21 s |
| 3 | Logistic Regression | 0.9941 | 0.9988 | 0.9874 | 0.9931 | 0.9969 | 1.93 s |

**Selected model: Random Forest** — ranked by F1-score, then recall, then
ROC-AUC.

Random Forest confusion matrix:

|  | Predicted legitimate | Predicted phishing |
|---|---:|---:|
| **Actual legitimate** | 26,945 (TN) | 25 (FP) |
| **Actual phishing** | 185 (FN) | 19,918 (TP) |

In plain language: out of 20,103 phishing URLs, **185 were missed**; out of
26,995 URLs flagged as phishing, **25 were false alarms**.

> These numbers describe this dataset only. Section 20 explains why real-world
> performance will be lower.

## 6. Technology stack

| Purpose | Tools |
|---|---|
| Language | Python 3.11+ (developed with 3.13) |
| Machine learning | Scikit-learn |
| Data processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Web application | Streamlit |
| Model persistence | Joblib |
| Testing | Pytest |
| Notebook | Jupyter / VS Code notebooks |

Tested with Pandas 3.1.0, NumPy 2.5.4, Scikit-learn 1.9.1 and Streamlit 1.67.0.
`requirements.txt` uses compatible version ranges.

## 7. Dataset

**PhiUSIIL Phishing URL Dataset** — UCI Machine Learning Repository, dataset 967.

- Source: <https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset>
- Paper: Prasad, A. & Chandra, S. (2024), *PhiUSIIL: A diverse security profile
  empowered phishing URL detection framework based on similarity index and
  lexical features*, Computers & Security.
- License: CC BY 4.0.
- Size used here: **235,795 labelled URLs** (134,850 legitimate, 100,945 phishing).
- Key point: the dataset ships **actual URL strings**, so this project computes
  its own features instead of relying on precomputed columns.

After cleaning: **235,362 URLs** (425 duplicate URLs and 8 malformed URLs removed).

### Label mapping

PhiUSIIL's original `label` column uses `1 = legitimate` and `0 = phishing`.
The project standardises everything to:

| Value | Meaning |
|---:|---|
| `1` | Phishing / malicious (positive class) |
| `0` | Legitimate / benign |

The conversion happens once in `src/data_loader.py` and is covered by a
regression test. See [`docs/dataset_guide.md`](docs/dataset_guide.md) for
download, placement and alternative-dataset instructions.

## 8. Feature engineering

`src/feature_extraction.py` exposes the single source of truth:

```python
extract_url_features(url: str) -> dict
```

The same function is used for training and for prediction, which prevents
train/serve skew. It uses only `urllib.parse` and regular expressions.

| Group | Features |
|---|---|
| Length and counts | `url_length`, `hostname_length`, `path_length`, `query_length`, `num_dots`, `num_hyphens`, `num_underscores`, `num_slashes`, `num_question_marks`, `num_equal_signs`, `num_at_symbols`, `num_ampersands`, `num_special_chars` |
| Character composition | `num_digits`, `digit_ratio`, `num_letters`, `letter_ratio` |
| Host structure | `num_subdomains`, `num_params`, `path_depth`, `tld_length`, `hostname_hyphen_count` |
| Transport / host type | `has_https`, `has_ip_address`, `has_unusual_port`, `has_port`, `has_punycode` |
| Risk signals | `has_suspicious_keyword`, `num_suspicious_keywords`, `has_url_shortener`, `has_brand_term`, `hostname_is_long`, `suspicious_tld` |
| Obfuscation | `num_percent_encoded`, `has_hex_encoding`, `has_double_slash_in_path`, `has_obfuscation`, `is_long_url` |

How some signals work:

- **Suspicious keywords** are matched as whole URL tokens (`verify`,
  `account`, `login`, `secure`, `wallet`, `webscr`, ...). `example.com/verify-account`
  matches; `example.com/accountant` does not.
- **Brand impostor** is `1` when a brand (`paypal`, `apple`, `microsoft`, ...)
  appears on a domain that is not that brand's official domain.
- **IP-address host** detects IPv4/IPv6 literals, including `[2001:db8::1]`.
- **Unusual port** flags explicit ports other than 80/443.
- **Suspicious TLD** uses a curated list of frequently abused TLDs (`tk`, `ml`,
  `xyz`, `top`, `click`, ...).
- **URL shortener** uses a fixed list (`bit.ly`, `tinyurl.com`, `t.co`, ...).

### Input validation and safety

- Empty, whitespace-containing, overly long (over 2,083 characters) and
  control-character-containing inputs are rejected.
- Only `http://` and `https://` are accepted; `javascript:`, `data:`, `file:`
  and similar schemes are rejected.
- A missing scheme is treated as `http://` (conservative for analysis).
- Hostnames are validated (labels, hyphens, TLD), including IPv6 literals.
- **The URL is never opened, fetched, mounted or executed. No HTTP request is
  made. No JavaScript runs. No reputation API is called.**

## 9. Exploratory data analysis

Executed notebook: [`notebooks/phishing_url_eda.ipynb`](notebooks/phishing_url_eda.ipynb)
(with outputs saved). Figures are written to `results/eda/`.

| Figure | What it shows |
|---|---|
| `class_distribution.png` | 134,850 legitimate vs 100,512 phishing (57.3% / 42.7%) |
| `length_distributions.png` | URL and hostname length by class |
| `special_character_analysis.png` | Mean special-character counts by class |
| `feature_comparison.png` | Box plots for six informative features |
| `correlation_heatmap.png` | Features most correlated with the label |
| `top_tlds.png` | Most frequent top-level domains |

Key EDA findings:

- Phishing URLs are longer on average (**45.9 vs 27.2** characters) and have
  longer hostnames (**24.4 vs 19.2**).
- Phishing URLs contain far more digits (**4.22 vs 0.05** on average) and
  hyphens (**0.69 vs 0.08**).
- The strongest single correlate of phishing is `has_https` (**-0.614**) —
  because **100% of legitimate URLs in this dataset use HTTPS** while only about
  half of phishing URLs do. This is a **dataset bias**, discussed in
  [Limitations](#20-limitations).
- `path_depth` is higher for phishing: most legitimate examples are homepages.
- Suspicious TLDs and IP-address hosts are rare but strongly phishing-associated.

To re-run the notebook:

```powershell
python -m src.preprocessing
jupyter nbconvert --to notebook --execute --inplace notebooks/phishing_url_eda.ipynb
```

Or open it in VS Code and run all cells.

## 10. Machine learning algorithms

| Model | Configuration | Why it is included |
|---|---|---|
| Logistic Regression | `StandardScaler` + `LogisticRegression(class_weight="balanced", max_iter=2000)` in a Pipeline | Fast, interpretable linear baseline |
| Decision Tree | `max_depth=14`, `min_samples_leaf=5`, balanced class weights | Non-linear model, easy to explain |
| Random Forest | 300 trees, `min_samples_leaf=2`, `max_features="sqrt"`, balanced subsample weights | Strong ensemble, robust to feature scaling and correlated features |

Important design choices:

- All models train on **exactly the same** `X_train, y_train` and are evaluated
  on the same `X_test, y_test`.
- `class_weight="balanced"` compensates for the mild 1.34:1 class imbalance.
- **Scaling is fitted inside the Logistic Regression pipeline**, so the test set
  never influences feature scaling. No scaling or feature selection touches the
  full dataset before splitting.
- `random_state=42` everywhere, with a stratified 80/20 split.

## 11. Evaluation methodology

The split is stratified, so both sets preserve the class ratio:

- Training rows: **188,289**
- Test rows: **47,073** (26,970 legitimate, 20,103 phishing)

Metrics are computed by `src/model_evaluation.py` from real predictions.

### Beginner-friendly metric explanations

- **Accuracy** — the share of all URLs classified correctly. Easy to
  understand, but misleading if one class dominates.
- **Precision** — of all URLs the model *flagged* as phishing, how many really
  were phishing?
  `precision = TP / (TP + FP)`.
  Low precision means many false alarms against legitimate sites.
- **Recall** (sensitivity) — of all *actual* phishing URLs, how many the model
  caught?
  `recall = TP / (TP + FN)`.
  Low recall means dangerous misses.
- **F1-score** — the harmonic mean of precision and recall. A single balanced
  number that punishes extreme trade-offs.
- **Confusion matrix** — a 2×2 table of true negatives, false positives, false
  negatives and true positives. It shows *which kind* of error the model makes.
- **ROC-AUC** — how well the model's probability score separates the classes
  across every possible threshold. `1.0` is perfect, `0.5` is random.

### Why precision and recall both matter

For phishing detection, **recall is usually prioritised**: missing a malicious
URL can lead to stolen credentials. But **precision matters too**: if the tool
cries wolf, users ignore it and legitimate businesses get blocked.

The project therefore ranks models by **F1-score first, then recall, then
ROC-AUC** — never by accuracy alone. Random Forest won with F1 0.9948 and recall
0.9908; Logistic Regression had slightly higher precision (0.9988) but lower
recall (0.9874).

Artifacts produced in `results/`:

```
model_comparison.csv / .json        metrics table for all models
evaluation_metrics.json             full metrics + confusion matrices + reports
training_summary.json               training metadata and selected model
metrics_<model>.json                per-model detailed metrics
classification_report_<model>.txt   per-model classification report
confusion_matrix_<model>.png        per-model confusion matrix
metrics_comparison.png              grouped metric bars
accuracy/precision/recall/f1_comparison.png
roc_curves.png                      ROC curves for all models
feature_importance.png              top Random Forest features
```

## 12. Project architecture

```
                 ┌──────────────────────────────┐
                 │  data/raw/urls.csv           │
                 │  (url, label)                │
                 └──────────────┬───────────────┘
                                │
                    src/data_loader.py
              (validate columns, normalise labels)
                                │
                  src/preprocessing.py
        (dedupe → validate → extract 38 features → cache)
                                │
        ┌───────────────────────┴────────────────────────┐
        │                                                │
  notebooks/phishing_url_eda.ipynb              src/model_training.py
      (EDA → results/eda/*.png)              (LR / DT / RF, stratified split)
                                                       │
                                             src/model_evaluation.py
                                          (metrics, charts, model selection)
                                                       │
                                  models/phishing_detector.joblib + metadata
                                                       │
                                              src/prediction.py
                                        (validate → features → score)
                                                       │
                                                    app.py
                                    (Streamlit: dashboard / predict / metrics)
```

Full workflow:

```
Dataset → Data Cleaning → Feature Extraction → EDA → Train/Test Split
        → Model Training → Evaluation → Model Saving → Streamlit Prediction
```

## 13. Folder structure

```
Phishing_URL_Detector/
├── app.py                          Streamlit application
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
├── pytest.ini
│
├── config/
│   ├── __init__.py
│   └── settings.py                 paths, constants, label mapping, app text
│
├── data/
│   ├── raw/                        urls.csv + downloaded PhiUSIIL files
│   └── processed/                  url_features.csv (cached feature matrix)
│
├── notebooks/
│   └── phishing_url_eda.ipynb      executed EDA notebook
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py              dataset discovery, loading, label normalisation
│   ├── preprocessing.py            cleaning, feature matrix, stratified split
│   ├── feature_extraction.py       38 URL features + validation
│   ├── model_training.py           train/compare/select/persist models
│   ├── model_evaluation.py         metrics, reports, charts
│   ├── prediction.py               load model, validate input, predict
│   └── utils.py                    logging, JSON, helpers
│
├── models/
│   ├── phishing_detector.joblib    selected model (Random Forest)
│   ├── phishing_detector_metadata.json
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   └── random_forest.joblib
│
├── results/
│   ├── eda/                        EDA figures
│   ├── *.png                       evaluation charts
│   ├── *.csv / *.json / *.txt      metrics and reports
│   └── .gitkeep
│
├── tests/
│   ├── conftest.py
│   ├── test_feature_extraction.py
│   ├── test_preprocessing.py
│   └── test_prediction.py
│
└── docs/
    ├── dataset_guide.md
    └── screenshots/README.md
```

## 14. Installation on Windows

Requirements: Windows 10/11, Python 3.11 or newer, and internet access for the
one-time dataset download.

Open **PowerShell** in the project root (the folder containing `app.py`).

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Activate it

```powershell
.venv\Scripts\activate
```

If PowerShell blocks the script, run this once for the current user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 3. Upgrade pip and install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 15. Dataset setup

### Option A — automatic download (recommended)

```powershell
python -m src.data_loader --download-phiusiil
```

This downloads the PhiUSIIL archive from UCI, extracts it to
`data/raw/phiusiil/`, converts the labels and writes the standardised file:

```
data/raw/urls.csv        columns: url,label   (1 = phishing, 0 = legitimate)
```

### Option B — manual placement

1. Download the dataset from
   <https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset>.
2. Unzip it.
3. Place `PhiUSIIL_Phishing_URL_Dataset.csv` in
   `data/raw/phiusiil/` **or** create your own `data/raw/urls.csv` with columns
   `url,label`.
4. Run:

```powershell
python -m src.data_loader --download-phiusiil
```

The command converts an already-extracted CSV without re-downloading it.

Full details, alternative datasets and label conventions:
[`docs/dataset_guide.md`](docs/dataset_guide.md).

> If you use another dataset, make sure it contains **actual URL strings**.
> A file with only precomputed numerical features cannot be used by this
> project's feature extractor.

## 16. Training and evaluation commands

All commands run from the project root.

```powershell
# 1. Clean URLs and extract the 38 features (~4 minutes for 235k URLs)
python -m src.preprocessing

# (optional) force a rebuild from the raw dataset
python -m src.preprocessing --force

# 2. Train the three models, select and save the best one (~1 minute)
python -m src.model_training

# 3. Generate reports and charts in results/
python -m src.model_evaluation
```

Useful options:

```powershell
python -m src.preprocessing --dataset path\to\other.csv --limit 5000
python -m src.model_training --test-size 0.25 --random-state 42
python -m src.model_training --force
```

Retrain from a clean checkout:

```powershell
python -m src.data_loader --download-phiusiil
python -m src.preprocessing --force
python -m src.model_training
python -m src.model_evaluation
```

## 17. Running the application

```powershell
streamlit run app.py
```

Open <http://localhost:8501>.

Pages:

1. **Dashboard** — introduction, model metrics, dataset facts, workflow.
2. **URL Prediction** — paste a URL, validate, extract features, predict, show
   probability, feature table and safety disclaimer.
3. **Model Performance** — metric table, confusion matrix, ROC curves,
   comparison charts and classification report.
4. **About** — objective, workflow, dataset source, features, models,
   limitations and safety notes.

Command-line prediction (no UI):

```powershell
python -m src.prediction "https://www.example.com/login"
python -m src.prediction "http://secure-paypal-login.verify-account.tk/update.php?id=99"
```

## 18. Running the tests

```powershell
pytest -v
```

Last local run: **57 passed**.

Coverage includes:

- feature schema, ordering and numeric consistency
- HTTPS detection, IP detection, length calculations, subdomain counting
- suspicious keyword, shortener, brand-impostor and port detection
- invalid/empty URL handling (no network calls are made)
- PhiUSIIL label inversion
- cleaning (missing, duplicate, invalid rows)
- stratified, reproducible splitting and disjoint train/test indices
- scaler placement inside the model pipeline
- model loading, feature-order validation, corrupted/missing model errors
- prediction output, probabilities, determinism and error collection

## 19. Screenshots

Run the app and add your own screenshots to `docs/screenshots/`:
`dashboard.png`, `prediction.png`, `performance.png`.

```markdown
![Dashboard](docs/screenshots/dashboard.png)
![Prediction](docs/screenshots/prediction.png)
```

No screenshots are bundled, so nothing in this repository is fabricated.

## 20. Limitations

1. **URL-only analysis.** The model sees only the URL string. It does not
   inspect page content, JavaScript, forms, certificates, redirects, WHOIS data,
   domain age or hosting reputation.
2. **Dataset bias — HTTPS.** In PhiUSIIL, **100% of legitimate URLs use HTTPS**
   while only ~49% of phishing URLs do. The model can therefore learn a shortcut
   ("HTTP is dangerous"). Never treat HTTPS as proof of safety.
3. **Path bias.** Legitimate examples are mostly homepages, while many phishing
   URLs contain paths. URLs such as `https://github.com/login` can be
   over-flagged even though the domain is legitimate.
4. **Feature-vector collisions.** Many different URLs share the same 38
   feature values. Random row-level splitting can therefore look optimistic
   compared with evaluation grouped by domain or time.
5. **Temporal drift.** Phishing campaigns change constantly; models need
   retraining on fresh data.
6. **Imbalanced error costs.** Reported test metrics use a 0.5 probability
   threshold. A deployment might tune the threshold toward higher recall.
7. **Not a security product.** False positives and false negatives are
   unavoidable. Use browser protection, multi-factor authentication and
   email/web filtering alongside this tool.
8. **No guarantee.** "Predicted as Legitimate" means "similar to legitimate
   examples in this dataset", not "verified safe".

## 21. Future improvements

- Group-aware or time-based train/test splits (by domain and collection date)
  to measure generalisation more honestly.
- Probability calibration (Platt/isotonic) and a configurable decision threshold.
- More features: entropy, longest token, consonant runs, punycode decoding,
  registrable-domain extraction with the Public Suffix List.
- Add a validated second dataset (e.g. PhishTank/OpenPhish snapshots) and test
  cross-dataset performance.
- Explain individual predictions with SHAP or per-feature contributions.
- Add batch CSV upload with downloadable predictions.
- Track experiments with MLflow and monitor data drift.
- Optional, explicitly configured reputation API integration (off by default).
- Docker packaging and CI (lint + tests) via GitHub Actions.

## 22. Troubleshooting

| Problem | Solution |
|---|---|
| `No dataset found` | Run `python -m src.data_loader --download-phiusiil`, or place your CSV at `data/raw/urls.csv`. |
| `No trained model found` | Run `python -m src.model_training`. |
| `The model file ... may be corrupted` | Run `python -m src.model_training --force`. |
| `feature order` error | Retrain with the current code: `python -m src.model_training --force`. |
| `streamlit: command not found` | Activate the virtual environment and reinstall requirements. |
| PowerShell refuses `.venv\Scripts\activate` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. |
| Dataset column error | Your CSV needs a URL column (`url`/`link`/`uri`/...) and a label column (`label`/`class`/`target`/...). |
| Very long preprocessing time | Expected (~4 minutes for 235k URLs). Use `--limit` for a quick test. |

## 23. Dataset attribution and license

- **Dataset:** PhiUSIIL Phishing URL Dataset.
- **Authors:** Arvind Prasad and Shalini Chandra.
- **Publisher:** UCI Machine Learning Repository, dataset ID 967.
- **URL:** <https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset>
- **License:** CC BY 4.0.
- **Related paper:** Prasad, A., Chandra, S. (2024). *PhiUSIIL: A diverse security
  profile empowered phishing URL detection framework based on similarity index
  and lexical features.* Computers & Security, 103545.

The dataset is redistributed here only in derived form (URL + normalised label,
plus engineered features). Cite the original authors if you reuse it.

## 24. Disclaimer

This software is provided for **educational and research purposes**. It performs
a static classification of a URL string and does not verify, visit or certify
websites. Predictions may be wrong in both directions. The authors accept no
liability for decisions made based on its output. Do not enter credentials or
payment information based on this tool's result.
