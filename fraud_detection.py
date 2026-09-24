# -*- coding: utf-8 -*-
"""

# -*- coding: utf-8 -*-
"""
COEN807 Term Project - Machine Learning for Real-World Data Analytics
Supervised Learning: Classification - Credit Card Fraud Detection

Author: Idris Muhammad Abubakar
Student ID: P25EGCP9003
Date: June, 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score, 
                             roc_curve, confusion_matrix, classification_report)
from imblearn.under_sampling import RandomUnderSampler
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD DATASET
# ============================================================
print("="*70)
print("COEN807 TERM PROJECT - CREDIT CARD FRAUD DETECTION")
print("="*70)

# Update this path to where your dataset is located
file_path = r'D:/i00816741/Desktop/PG ABU/MPhil/2026/1st Semester Lectures/COEN 807 Machine Learning/Assignment/Term Project/creditcard.csv'

df = pd.read_csv(file_path)
X = df.drop('Class', axis=1)
y = df['Class']

print(f"\nDataset shape: {df.shape}")
print(f"Fraud cases: {df['Class'].sum()}")
print(f"Legitimate cases: {len(df) - df['Class'].sum()}")
print(f"Fraud percentage: {df['Class'].mean()*100:.4f}%")

# ============================================================
# 2. TRAIN-TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# ============================================================
# 3. HANDLE CLASS IMBALANCE (UNDERSAMPLING)
# ============================================================
rus = RandomUnderSampler(random_state=42)
X_train_balanced, y_train_balanced = rus.fit_resample(X_train, y_train)
print(f"\nBalanced training set size: {len(X_train_balanced)}")
print(f"Fraud cases in balanced set: {sum(y_train_balanced)}")
print(f"Legitimate cases in balanced set: {len(y_train_balanced) - sum(y_train_balanced)}")

# ============================================================
# 4. FEATURE SCALING
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)
print("\nFeature scaling complete.")

# ============================================================
# 5. MODEL 1: LOGISTIC REGRESSION
# ============================================================
print("\n" + "="*70)
print("MODEL 1: LOGISTIC REGRESSION")
print("="*70)

start_time = time.time()
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train_balanced)
lr_time = time.time() - start_time

y_pred_lr = lr.predict(X_test_scaled)
y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

print(f"Training Time: {lr_time:.2f} seconds")
print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_lr):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_lr):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba_lr):.4f}")

# ============================================================
# 6. MODEL 2: DECISION TREE (with hyperparameter tuning)
# ============================================================
print("\n" + "="*70)
print("MODEL 2: DECISION TREE")
print("="*70)

param_grid_dt = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

start_time = time.time()
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5, scoring='f1', n_jobs=-1)
dt_grid.fit(X_train_scaled, y_train_balanced)
dt_time = time.time() - start_time

best_dt = dt_grid.best_estimator_
print(f"Best parameters: {dt_grid.best_params_}")
print(f"Tuning Time: {dt_time:.2f} seconds")

y_pred_dt = best_dt.predict(X_test_scaled)
y_proba_dt = best_dt.predict_proba(X_test_scaled)[:, 1]

print(f"Precision: {precision_score(y_test, y_pred_dt):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_dt):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_dt):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba_dt):.4f}")

# ============================================================
# 7. MODEL 3: SVM (with hyperparameter tuning)
# ============================================================
print("\n" + "="*70)
print("MODEL 3: SUPPORT VECTOR MACHINE")
print("="*70)

param_grid_svm = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]}

start_time = time.time()
svm_grid = GridSearchCV(SVC(kernel='rbf', random_state=42, probability=True), 
                         param_grid_svm, cv=5, scoring='f1', n_jobs=-1)
svm_grid.fit(X_train_scaled, y_train_balanced)
svm_time = time.time() - start_time

best_svm = svm_grid.best_estimator_
print(f"Best parameters: C={svm_grid.best_params_['C']}, gamma={svm_grid.best_params_['gamma']}")
print(f"Tuning Time: {svm_time:.2f} seconds")

y_pred_svm = best_svm.predict(X_test_scaled)
y_proba_svm = best_svm.predict_proba(X_test_scaled)[:, 1]

print(f"Precision: {precision_score(y_test, y_pred_svm):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_svm):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_svm):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba_svm):.4f}")

# ============================================================
# 8. COMPARISON SUMMARY TABLE
# ============================================================
print("\n" + "="*70)
print("SUMMARY: MODEL COMPARISON")
print("="*70)

summary_data = {
    'Model': ['Logistic Regression', 'Decision Tree', 'SVM'],
    'Precision': [
        precision_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_dt),
        precision_score(y_test, y_pred_svm)
    ],
    'Recall': [
        recall_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_dt),
        recall_score(y_test, y_pred_svm)
    ],
    'F1-Score': [
        f1_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_dt),
        f1_score(y_test, y_pred_svm)
    ],
    'AUC-ROC': [
        roc_auc_score(y_test, y_proba_lr),
        roc_auc_score(y_test, y_proba_dt),
        roc_auc_score(y_test, y_proba_svm)
    ],
    'Tuning_Time_sec': [lr_time, dt_time, svm_time]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# ============================================================
# 9. VISUALIZATION 1: ROC CURVES COMPARISON
# ============================================================
plt.figure(figsize=(10, 8))

# ROC Curves
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_proba_dt)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_proba_svm)

plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, y_proba_lr):.4f})', linewidth=2)
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {roc_auc_score(y_test, y_proba_dt):.4f})', linewidth=2)
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {roc_auc_score(y_test, y_proba_svm):.4f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)', linewidth=1)

plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Recall)', fontsize=12)
plt.title('ROC Curves: Model Comparison for Credit Card Fraud Detection', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.savefig('model_comparison_roc.png', dpi=300, bbox_inches='tight')
print("\n✓ Figure saved: model_comparison_roc.png")
plt.show()

# ============================================================
# 10. VISUALIZATION 2: PERFORMANCE BAR CHART
# ============================================================
metrics = ['Precision', 'Recall', 'F1-Score', 'AUC-ROC']
lr_scores = [precision_score(y_test, y_pred_lr), recall_score(y_test, y_pred_lr), 
             f1_score(y_test, y_pred_lr), roc_auc_score(y_test, y_proba_lr)]
dt_scores = [precision_score(y_test, y_pred_dt), recall_score(y_test, y_pred_dt), 
             f1_score(y_test, y_pred_dt), roc_auc_score(y_test, y_proba_dt)]
svm_scores = [precision_score(y_test, y_pred_svm), recall_score(y_test, y_pred_svm), 
              f1_score(y_test, y_pred_svm), roc_auc_score(y_test, y_proba_svm)]

x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width, lr_scores, width, label='Logistic Regression', color='blue')
rects2 = ax.bar(x, dt_scores, width, label='Decision Tree', color='green')
rects3 = ax.bar(x + width, svm_scores, width, label='SVM', color='orange')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim(0, 1.1)

# Add value labels on bars
for rects in [rects1, rects2, rects3]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('model_comparison_barchart.png', dpi=300, bbox_inches='tight')
print("✓ Figure saved: model_comparison_barchart.png")
plt.show()

# ============================================================
# 11. CONFUSION MATRICES FOR ALL MODELS
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

models = [('Logistic Regression', y_pred_lr), ('Decision Tree', y_pred_dt), ('SVM', y_pred_svm)]

for idx, (name, y_pred) in enumerate(models):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Legitimate', 'Fraud'], yticklabels=['Legitimate', 'Fraud'])
    axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Predicted Label')
    axes[idx].set_ylabel('True Label')

plt.suptitle('Confusion Matrices: Model Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('model_comparison_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Figure saved: model_comparison_confusion_matrices.png")
plt.show()

# ============================================================
# 12. CROSS-VALIDATION RESULTS
# ============================================================
print("\n" + "="*70)
print("CROSS-VALIDATION RESULTS (5-FOLD CV ON TRAINING SET)")
print("="*70)

cv_lr = cross_val_score(LogisticRegression(random_state=42, max_iter=1000), 
                         X_train_scaled, y_train_balanced, cv=5, scoring='f1')
cv_dt = cross_val_score(DecisionTreeClassifier(**dt_grid.best_params_, random_state=42), 
                         X_train_scaled, y_train_balanced, cv=5, scoring='f1')
cv_svm = cross_val_score(SVC(kernel='rbf', C=svm_grid.best_params_['C'], 
                              gamma=svm_grid.best_params_['gamma'], random_state=42), 
                         X_train_scaled, y_train_balanced, cv=5, scoring='f1')

print(f"Logistic Regression CV F1: {cv_lr.mean():.4f} (+/- {cv_lr.std():.4f})")
print(f"Decision Tree CV F1: {cv_dt.mean():.4f} (+/- {cv_dt.std():.4f})")
print(f"SVM CV F1: {cv_svm.mean():.4f} (+/- {cv_svm.std():.4f})")

print("\n" + "="*70)
print("PROJECT COMPLETE")
print("="*70)
