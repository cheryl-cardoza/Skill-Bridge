"""
ALL-IN-ONE SETUP SCRIPT
=======================
Run this ONCE from inside your student_project/ folder:
    python setup.py

It will:
  1. Train all 4 models on your StudentsPerformance.csv
  2. Save student_bundle.pkl locally (trained on YOUR machine = safe)
  3. Generate all 6 chart images into static/charts/

Requirements:
    pip install flask scikit-learn numpy pandas matplotlib seaborn
"""

import os, pickle, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

BASE    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, 'static', 'charts')
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════
# STEP 1 — Load your CSV
# ══════════════════════════════════════════════════════
csv_path = os.path.join(BASE, 'StudentsPerformance.csv')
if not os.path.exists(csv_path):
    raise FileNotFoundError(
        "\n\nCannot find StudentsPerformance.csv\n"
        "Make sure it is in the same folder as this script.\n"
    )

df = pd.read_csv(csv_path)
df.columns = (df.columns.str.strip().str.lower()
              .str.replace(' ', '_').str.replace('/', '_'))
print(f"Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")
print("Columns:", df.columns.tolist())

# ══════════════════════════════════════════════════════
# STEP 2 — Encode categorical columns
# ══════════════════════════════════════════════════════
df_enc   = df.copy()
encoders = {}
cat_cols = ['gender', 'race_ethnicity', 'parental_level_of_education',
            'lunch', 'test_preparation_course']

for col in cat_cols:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col])
    encoders[col] = le

X = df_enc.drop('math_score', axis=1)
y = df_enc['math_score']
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ══════════════════════════════════════════════════════
# STEP 3 — Train all 4 models
# ══════════════════════════════════════════════════════
print("\nTraining models with hyperparameter tuning...")
import time
from sklearn.model_selection import GridSearchCV

# Tune Random Forest — smaller grid to avoid RAM crash
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
}
print("  Tuning Random Forest (may take ~60s, single-threaded to save RAM)...")
t = time.time()
rf_gs = GridSearchCV(
    RandomForestRegressor(random_state=42),
    rf_params, cv=3, scoring='r2', n_jobs=1)   # n_jobs=1 = no parallel, safe on low RAM
rf_gs.fit(X_train, y_train)
print(f"  Best RF params: {rf_gs.best_params_}  ({time.time()-t:.1f}s)")

# Tune Gradient Boosting — smaller grid
gb_params = {
    'n_estimators': [100],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
}
print("  Tuning Gradient Boosting (may take ~20s)...")
t = time.time()
gb_gs = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    gb_params, cv=3, scoring='r2', n_jobs=1)   # n_jobs=1 = safe on low RAM
gb_gs.fit(X_train, y_train)
print(f"  Best GB params: {gb_gs.best_params_}  ({time.time()-t:.1f}s)")

models_def = {
    'Linear Regression':  LinearRegression(),
    'Ridge Regression':   Ridge(alpha=1.0),
    'Random Forest':      rf_gs.best_estimator_,
    'Gradient Boosting':  gb_gs.best_estimator_,
}
tuning_params = {
    'Random Forest':     rf_gs.best_params_,
    'Gradient Boosting': gb_gs.best_params_,
}

results        = {}
trained_models = {}

for name, model in models_def.items():
    if name not in ('Random Forest', 'Gradient Boosting'):
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    cv   = cross_val_score(model, X, y, cv=3, scoring='r2', n_jobs=1).mean()  # n_jobs=1
    results[name]        = {'R2': round(r2,4), 'RMSE': round(rmse,4), 'CV_R2': round(cv,4)}
    trained_models[name] = model
    print(f"  {name:25s}  R²={r2:.3f}  RMSE={rmse:.2f}  CV_R²={cv:.3f}")

best_model = max(results, key=lambda k: results[k]['R2'])
print(f"\nBest model: {best_model}  (R²={results[best_model]['R2']})")

# ══════════════════════════════════════════════════════
# STEP 4 — Save bundle (trained locally = safe)
# ══════════════════════════════════════════════════════
bundle = {
    'models':        trained_models,
    'best_model':    best_model,
    'encoders':      encoders,
    'feature_names': feature_names,
    'results':       results,
    'tuning_params': tuning_params,
}
pkl_path = os.path.join(BASE, 'student_bundle.pkl')
with open(pkl_path, 'wb') as f:
    pickle.dump(bundle, f)
print(f"\nSaved: student_bundle.pkl")

# ══════════════════════════════════════════════════════
# STEP 5 — Generate all charts (one at a time to save RAM)
# ══════════════════════════════════════════════════════
print("\nGenerating charts...")
import gc
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10})
matplotlib.rcParams['figure.max_open_warning'] = 1

def save(filename):
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=72, bbox_inches='tight', facecolor='white')
    plt.close('all')   # close ALL figures, not just current
    gc.collect()       # free memory immediately
    print(f"  saved -> static/charts/{filename}")

# Chart 1 — Model Comparison
fig, ax = plt.subplots(figsize=(5, 3), facecolor='white')
names  = list(results.keys())
r2vals = [results[n]['R2'] for n in names]
bars   = ax.barh(names, r2vals,
                 color=['#6366f1','#10b981','#f59e0b','#ef4444'],
                 height=0.5, edgecolor='none')
for bar, val in zip(bars, r2vals):
    ax.text(val - 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', ha='right',
            color='white', fontsize=9, fontweight='bold')
ax.set_xlim(0, 1.05)
ax.set_xlabel('R² Score')
ax.set_title('Model Comparison — R² Score', fontsize=11, fontweight='bold')
ax.spines[['top','right','left']].set_visible(False)
ax.tick_params(left=False)
plt.tight_layout()
save('model_comparison.png')

# Chart 2 — Feature Importance
rf          = trained_models['Random Forest']
importances = rf.feature_importances_
feat_labels = ['Gender','Race','Parent Edu','Lunch','Test Prep','Reading','Writing']
sorted_idx  = np.argsort(importances)
fig, ax = plt.subplots(figsize=(5, 3), facecolor='white')
bar_colors = ['#6366f1' if feat_labels[i] in ('Reading','Writing') else '#94a3b8'
              for i in sorted_idx]
ax.barh([feat_labels[i] for i in sorted_idx], importances[sorted_idx],
        color=bar_colors, edgecolor='none', height=0.55)
ax.set_title('Feature Importance (Random Forest)', fontsize=11, fontweight='bold')
ax.set_xlabel('Importance')
ax.spines[['top','right','left']].set_visible(False)
ax.tick_params(left=False)
plt.tight_layout()
save('feature_importance.png')

# Chart 3 — Score Distributions (one subplot at a time, then combined)
for col, color, label in [
    ('reading_score','#6366f1','Reading'),
    ('writing_score','#10b981','Writing'),
    ('math_score',  '#f59e0b','Math'),
]:
    fig, ax = plt.subplots(figsize=(4, 2.5), facecolor='white')
    ax.hist(df[col], bins=15, color=color, edgecolor='white', alpha=0.85)
    ax.axvline(df[col].mean(), color='#1f2937', linestyle='--', linewidth=1.2)
    ax.set_title(f'{label} Score  (mean={df[col].mean():.1f})', fontsize=10, fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    save(f'dist_{label.lower()}.png')

# Distributions combined — small size
fig, axes = plt.subplots(1, 3, figsize=(8, 2.5), facecolor='white')
for ax, (col, color, label) in zip(axes, [
    ('reading_score','#6366f1','Reading'),
    ('writing_score','#10b981','Writing'),
    ('math_score',  '#f59e0b','Math'),
]):
    ax.hist(df[col], bins=15, color=color, edgecolor='white', alpha=0.85)
    ax.axvline(df[col].mean(), color='#1f2937', linestyle='--', linewidth=1.2)
    ax.set_title(f'{label} (μ={df[col].mean():.0f})', fontsize=9, fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
plt.suptitle('Score Distributions', fontsize=10, fontweight='bold')
plt.tight_layout()
save('distributions.png')

# Chart 4 — Correlation Heatmap
fig, ax = plt.subplots(figsize=(5, 4), facecolor='white')
corr_cols   = ['reading_score','writing_score','math_score',
               'parental_level_of_education','lunch','test_preparation_course']
corr_labels = ['Reading','Writing','Math','Parent Edu','Lunch','Test Prep']
corr = df_enc[corr_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            xticklabels=corr_labels, yticklabels=corr_labels,
            ax=ax, linewidths=0.5, square=True,
            annot_kws={'size': 8}, cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Heatmap', fontsize=11, fontweight='bold')
plt.tight_layout()
save('correlation.png')

# Chart 5 — Test Prep Effect
fig, ax = plt.subplots(figsize=(4, 2.5), facecolor='white')
prep = df.groupby('test_preparation_course')['math_score'].mean()
ax.bar(prep.index, prep.values,
       color=['#94a3b8','#6366f1'], width=0.5, edgecolor='none')
for i, (idx, val) in enumerate(prep.items()):
    ax.text(i, val + 0.4, f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')
ax.set_title('Test Prep vs Math Score', fontsize=11, fontweight='bold')
ax.set_ylabel('Avg Math Score')
ax.spines[['top','right','left']].set_visible(False)
plt.tight_layout()
save('test_prep.png')

# Chart 6 — Gender Breakdown
fig, ax = plt.subplots(figsize=(4, 2.5), facecolor='white')
gen = df.groupby('gender')['math_score'].mean()
ax.bar(gen.index, gen.values,
       color=['#6366f1','#10b981'], width=0.5, edgecolor='none')
for i, (idx, val) in enumerate(gen.items()):
    ax.text(i, val + 0.4, f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')
ax.set_title('Gender vs Math Score', fontsize=11, fontweight='bold')
ax.set_ylabel('Avg Math Score')
ax.spines[['top','right','left']].set_visible(False)
plt.tight_layout()
save('gender.png')

# ══════════════════════════════════════════════════════
print("\n[OK] Setup complete! Now run:  python app.py")
print("  Then open:  http://localhost:5000")
