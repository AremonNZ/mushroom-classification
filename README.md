mushroom-classification
ML classification project comparing KNN vs Decision Tree on 61K mushroom samples using scikit-learn

🍄 Mushroom Edibility Classification

A machine learning project to classify mushrooms as **edible or poisonous** using the UCI Secondary Mushroom Dataset. Two classification algorithms — K-Nearest Neighbours (KNN) and Decision Tree — are implemented, evaluated, and compared.

---

📁 Dataset

- **Source:** [UCI Machine Learning Repository – Secondary Mushroom Dataset](https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset)
- **Size:** 61,069 samples × 20 features
- **Target:** Binary classification — `p` (poisonous) or `e` (edible)
- **Features:** Cap diameter, cap shape, cap surface, gill colour, stem width, ring type, and more

---

🔍 Project Overview

 1. Data Exploration
- Loaded dataset directly via `ucimlrepo`
- Inspected shape, column types, and class distribution
- Identified missing values in several columns (e.g., `gill-spacing`, `cap-surface`)
- Visualised class distribution and categorical feature distributions using bar plots
- Generated a feature correlation heatmap

 2. Preprocessing
- Encoded all categorical features using `LabelEncoder`
- Scaled numerical features using `StandardScaler`
- Split data: **80% training / 20% test** (stratified)

3. Models

#### K-Nearest Neighbours (KNN)
- Trained with `k=5`
- Evaluated with classification report and confusion matrix
- Tested accuracy across `k = 1 to 20` to find optimal k

Decision Tree
- Trained a default (unpruned) Decision Tree
- Tuned with `max_depth=5` and `min_samples_split=10`
- Visualised feature importances

 4. Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| KNN (k=5) | 99.98% | 99.96% | 100% | 99.98% |
| Decision Tree (tuned) | 69.60% | 70.46% | 77.87% | 73.98% |

> KNN significantly outperformed the tuned Decision Tree on this dataset.

---

🛠️ Technologies Used

- Python 3
- Pandas, NumPy
- Scikit-learn (KNN, Decision Tree, StandardScaler, LabelEncoder)
- Matplotlib, Seaborn
- ucimlrepo

 📊 Key Findings

- KNN achieved near-perfect accuracy (99.98%) on the test set, making it highly effective for this dataset
- The tuned Decision Tree struggled more with precision/recall trade-offs but revealed useful feature importances
- `stem-width`, `stem-surface`, and `veil-color` were among the most important features according to the Decision Tree
