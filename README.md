# 🍄 Mushroom Edibility Classification

A machine learning project classifying mushrooms as **edible or poisonous** using the
real UCI Secondary Mushroom Dataset. Two classification algorithms — K-Nearest
Neighbours (KNN) and Decision Tree — are implemented, evaluated, and compared.

---

## 📁 Dataset

- **Source:** [UCI Machine Learning Repository — Secondary Mushroom Dataset](https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset)
- **Size:** 61,069 samples × 20 features
- **Target:** Binary classification — `e` (edible) or `p` (poisonous)

**Note on data ordering:** this dataset ships in two row orders — one generated
"ordered by species" and one "randomly shuffled." This project uses the shuffled
release (`secondary_data_shuffled.csv`) — see **Section: Debugging Note** below for why
that choice matters.

---

## 🔍 Project Workflow

1. **Data Exploration** — shape, column types, class distribution, missing values,
   duplicate check, categorical feature distributions, correlation heatmap
2. **Preprocessing** — label-encoded categorical features, split into train/test
   *before* fitting `StandardScaler` (fit on training data only, to avoid leakage)
3. **Model 1 — KNN** — trained at k=5, evaluated with a classification report and
   confusion matrix; swept k=1–20 to find the optimum
4. **Model 2 — Decision Tree** — trained both a default (unpruned) tree and a tuned
   version (`max_depth=5`, `min_samples_split=10`), compared, and visualised feature
   importances
5. **Cross-Validation** — 5-fold `StratifiedKFold` with `shuffle=True`, used to confirm
   the single train/test split result generalises

---

## 🐛 Debugging Note — a real inconsistency I found and fixed

An earlier version of this notebook fetched the dataset via the `ucimlrepo` package,
which returns the **species-ordered** release of this data (61,069 rows made up of
173 species in contiguous 353-row blocks). Running 5-fold cross-validation on that
ordering with `shuffle=False` (the default) put entire species blocks into single
folds — meaning each fold's test set could contain species the model had never seen
anything similar to during training on that fold. The result: single train/test split
accuracy of ~99.9%, but cross-validation accuracy of only ~52% (essentially chance) —
a huge, unexplained gap.

I traced this back to the dataset's row ordering rather than a real generalisation
failure, switched to the officially shuffled release of the same data
(`secondary_data_shuffled.csv`), and used an explicit `StratifiedKFold(shuffle=True)`.
That resolved the discrepancy — cross-validation accuracy now closely matches the
single-split result (see Key Findings below), confirming the original ~99.9% accuracy
was genuine and not an artefact of a lucky split.

---

## 🛠️ Technologies Used

Python 3 · Pandas, NumPy · Scikit-learn (KNN, Decision Tree, StandardScaler,
LabelEncoder, StratifiedKFold) · Matplotlib, Seaborn

---

## 📊 Key Findings

| Model | Test Accuracy | 5-Fold CV Accuracy |
|---|---|---|
| KNN (k=5) | 99.93% | 99.89% ± 0.03% |
| Decision Tree (default) | 99.85% | — |
| Decision Tree (tuned, max_depth=5) | 67.13% | 66.99% ± 0.31% |

- KNN and the unpruned Decision Tree both perform very well on this dataset — its 20
  features (particularly stem and cap measurements) separate the two classes cleanly.
- Once cross-validation was run correctly (shuffled data, shuffled folds), CV results
  closely track the single-split results for both models — the earlier apparent
  instability was a data-ordering bug, not a real modelling issue.
- The tuned (max_depth=5) Decision Tree trades a substantial amount of accuracy for
  interpretability compared to the unpruned tree and KNN — depth-5 is too shallow to
  capture this dataset's structure, which is a useful illustration of the
  accuracy/interpretability trade-off in tree pruning.

---

## 👤 Author

**Arman Arabkhani**
Final-year Computer Science (Data Science) student @ Auckland University of Technology
📧 armanarabkhani.nz@yahoo.com
🔗 [LinkedIn](https://www.linkedin.com/in/arman-arabkhani-95903a384/) | [GitHub](https://github.com/AremonNZ)
