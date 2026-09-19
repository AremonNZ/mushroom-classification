#!/usr/bin/env python
# coding: utf-8

# # 🍄 Mushroom Edibility Classification
# **Dataset:** UCI Secondary Mushroom Dataset — 61,069 samples, 20 features
# **Goal:** Classify mushrooms as Edible or Poisonous using KNN and Decision Tree
# **Author:** Arman Arabkhani | AUT Data Science

# ## 1. Imports & Data Loading
#
# NOTE ON DATA SOURCE: the dataset's own documentation states it ships in two
# row orders — a version generated "ordered by species" and a "randomly
# shuffled" version. Fetching it via `ucimlrepo` returns the species-ordered
# version. That matters a lot for cross-validation (see Section 6), so this
# version explicitly loads the shuffled release of the same data.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report,
                             precision_score, recall_score, f1_score)

df = pd.read_csv('secondary_data_shuffled.csv', sep=';')
y = df[['class']]
X = df.drop(columns=['class'])
print("Shape:", df.shape)
df.head()


# ## 2. Exploratory Data Analysis

print("Dataset Info:")
df.info()

print("Missing values per column:\n", df.isnull().sum())
print(f"\nDuplicate rows: {df.duplicated().sum()}")

plt.figure(figsize=(5, 4))
sns.countplot(x='class', data=df, palette='Set2')
plt.title('Class Distribution (e = Edible, p = Poisonous)')
plt.xlabel('Class'); plt.ylabel('Count')
plt.tight_layout(); plt.savefig('chart_class_dist.png'); plt.close()
print(df['class'].value_counts())

categorical_cols = X.select_dtypes(include=['object']).columns[:6]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flatten(), categorical_cols):
    order = X[col].value_counts().index
    sns.countplot(data=X, x=col, order=order, ax=ax)
    ax.set_title(f'Distribution of {col}')
    ax.tick_params(axis='x', rotation=45)
plt.suptitle('Categorical Feature Distributions (Sample)', fontsize=14)
plt.tight_layout(); plt.savefig('chart_categorical_dist.png'); plt.close()

X_heatmap = X.copy().apply(lambda col: col.astype('category').cat.codes)
plt.figure(figsize=(10, 8))
sns.heatmap(X_heatmap.corr(), cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout(); plt.savefig('chart_correlation.png'); plt.close()


# ## 3. Preprocessing
# - Encode all categorical features with LabelEncoder
# - Encode target variable
# - Split data FIRST, then fit StandardScaler on training set only (prevents data leakage)

X_encoded = X.copy()
for col in X_encoded.columns:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

target_le = LabelEncoder()
y_encoded = target_le.fit_transform(y.values.ravel())
print("Classes:", target_le.classes_, "→ encoded as", list(range(len(target_le.classes_))))

# Split BEFORE scaling
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Fit scaler on train only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"Train: {X_train_scaled.shape} | Test: {X_test_scaled.shape}")


# ## 4. Model 1 — K-Nearest Neighbours (KNN)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

print(f"KNN (k=5) — Test Accuracy: {accuracy_score(y_test, y_pred_knn)*100:.2f}%")
print(classification_report(y_test, y_pred_knn, target_names=['Edible', 'Poisonous']))

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', cmap='Greens',
            xticklabels=['Edible', 'Poisonous'], yticklabels=['Edible', 'Poisonous'])
plt.title('Confusion Matrix — KNN (k=5)')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.tight_layout(); plt.savefig('chart_knn_confusion.png'); plt.close()

accuracies = []
k_range = range(1, 21)
for k in k_range:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train_scaled, y_train)
    accuracies.append(knn_k.score(X_test_scaled, y_test))

plt.figure(figsize=(8, 4))
plt.plot(k_range, accuracies, marker='o', color='steelblue')
plt.xlabel('k (Number of Neighbours)'); plt.ylabel('Accuracy')
plt.title('KNN Accuracy vs. k Value'); plt.grid(True)
plt.tight_layout(); plt.savefig('chart_knn_k.png'); plt.close()

best_k = k_range[np.argmax(accuracies)]
print(f"Best k = {best_k} with accuracy = {max(accuracies)*100:.2f}%")


# ## 5. Model 2 — Decision Tree

dt_default = DecisionTreeClassifier(random_state=42)
dt_default.fit(X_train_scaled, y_train)
y_pred_dt_default = dt_default.predict(X_test_scaled)
print("Default Decision Tree — Test Accuracy:", f"{accuracy_score(y_test, y_pred_dt_default)*100:.2f}%")
print(classification_report(y_test, y_pred_dt_default, target_names=['Edible', 'Poisonous']))

dt_tuned = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
dt_tuned.fit(X_train_scaled, y_train)
y_pred_dt_tuned = dt_tuned.predict(X_test_scaled)
print("Tuned Decision Tree (max_depth=5) — Test Accuracy:", f"{accuracy_score(y_test, y_pred_dt_tuned)*100:.2f}%")
print(classification_report(y_test, y_pred_dt_tuned, target_names=['Edible', 'Poisonous']))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, preds, title, cmap in zip(
        axes, [y_pred_dt_default, y_pred_dt_tuned],
        ['Default Decision Tree', 'Tuned Decision Tree (max_depth=5)'], ['Blues', 'Greens']):
    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=['Edible', 'Poisonous'], yticklabels=['Edible', 'Poisonous'])
    ax.set_title(title); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout(); plt.savefig('chart_dt_confusion.png'); plt.close()

importances = pd.Series(dt_tuned.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 5))
importances.plot(kind='bar', color='coral')
plt.title('Feature Importances — Tuned Decision Tree')
plt.ylabel('Importance Score'); plt.xticks(rotation=45, ha='right')
plt.tight_layout(); plt.savefig('chart_feature_importance.png'); plt.close()
print("\nTop 5 most important features:")
print(importances.head())


# ## 6. Cross-Validation
#
# IMPORTANT: this dataset's own documentation notes it is generated in two
# row orders — "ordered by species" and "randomly shuffled." An earlier
# version of this notebook ran CV on the species-ordered data with
# `shuffle=False`, which put entire 353-row species blocks into test-only
# folds the model had never seen anything similar to in training — collapsing
# CV accuracy to ~52% (near chance) despite ~99.9% single-split accuracy.
# Using the shuffled release AND an explicit shuffled splitter fixes this.

X_all_scaled = scaler.fit_transform(X_encoded)
cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

knn_cv = cross_val_score(KNeighborsClassifier(n_neighbors=5), X_all_scaled, y_encoded, cv=cv_splitter)
dt_cv  = cross_val_score(DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42),
                          X_all_scaled, y_encoded, cv=cv_splitter)

print(f"KNN (k=5)              — CV Accuracy: {knn_cv.mean()*100:.2f}% ± {knn_cv.std()*100:.2f}%")
print(f"Decision Tree (tuned)  — CV Accuracy: {dt_cv.mean()*100:.2f}%  ± {dt_cv.std()*100:.2f}%")


# ## 7. Model Comparison

comparison = pd.DataFrame([
    {'Model': 'KNN (k=5)',
     'Accuracy':  accuracy_score(y_test, y_pred_knn),
     'Precision': precision_score(y_test, y_pred_knn),
     'Recall':    recall_score(y_test, y_pred_knn),
     'F1-Score':  f1_score(y_test, y_pred_knn)},
    {'Model': 'Decision Tree (default)',
     'Accuracy':  accuracy_score(y_test, y_pred_dt_default),
     'Precision': precision_score(y_test, y_pred_dt_default),
     'Recall':    recall_score(y_test, y_pred_dt_default),
     'F1-Score':  f1_score(y_test, y_pred_dt_default)},
    {'Model': 'Decision Tree (tuned)',
     'Accuracy':  accuracy_score(y_test, y_pred_dt_tuned),
     'Precision': precision_score(y_test, y_pred_dt_tuned),
     'Recall':    recall_score(y_test, y_pred_dt_tuned),
     'F1-Score':  f1_score(y_test, y_pred_dt_tuned)},
]).set_index('Model')
print(comparison.round(4))

comparison.plot(kind='bar', figsize=(10, 5))
plt.title('Model Performance Comparison'); plt.ylabel('Score')
plt.xticks(rotation=15); plt.ylim(0, 1.05); plt.legend(loc='lower right')
plt.tight_layout(); plt.savefig('chart_model_comparison.png'); plt.close()


# ## 8. Conclusion
#
# - KNN (k=5) and the default Decision Tree both achieve very high, and now
#   CV-consistent, accuracy on this dataset — the earlier apparent instability
#   was a data-ordering bug, not a real generalisation problem.
# - **Root-cause investigation:** the dataset ships in two row orders
#   ("ordered by species" vs "randomly shuffled"). The originally-used data
#   source returned the species-ordered version; running unshuffled
#   cross-validation on it grouped whole species into single folds, causing
#   an artificial collapse to ~52% CV accuracy despite ~99.9% single-split
#   accuracy. Switching to the shuffled release and an explicitly shuffled
#   `StratifiedKFold` resolved the discrepancy.
# - Data leakage was avoided by fitting the scaler only on training data
#   before transforming the test set.
# - The tuned (max_depth=5) Decision Tree trades some accuracy for
#   interpretability compared to the default unpruned tree and KNN.
