import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import shap
from sklearn.inspection import permutation_importance
import warnings

# Suppress warnings to prevent unnecessary messages from being displayed
# Enablling a specific pandas option to control silent downcasting behavior in some versions
warnings.filterwarnings('ignore')
pd.set_option('future.no_silent_downcasting', True)

# Setting the  environment for parallel processing
os.environ["LOKY_MAX_CPU_COUNT"] = "2"


# Loading the datasets
try:
    data1 = pd.read_csv('pone.0304132.s001.csv')  
    data2 = pd.read_csv('cleaned_pone.0304132.s002.csv')
    data3 = pd.read_csv('Student Mental health.csv')
except Exception as e:
    print("[ERROR] Failed to load datasets:", e)
    exit()

# Standardizing the column names 
for df in [data1, data2, data3]:
    df.columns = df.columns.str.lower().str.strip()

# Renaming the columns to attain consistency
rename_mapping = {
    "do you have depression?": "depressed",
    "dep": "depressed",
    "do you have anxiety?": "anxiety",
    "suicide": "suicidal",
    "deptype": "depression_type",
    "do you have panic attack?": "panic_attack"
}

for df in [data1, data2, data3]:
    df.rename(columns=rename_mapping, inplace=True)

# Merging the datasets to have a combined data
combined_data = pd.concat([data1, data2, data3], axis=0, ignore_index=True)

if combined_data.empty:
    print("[ERROR] Merged dataset is empty! Kindly Check your sources.")
    exit()


# we are defining the target variable we want to analyse
target_variable = "depressed"
# checking if the target variable is in the dataset
if target_variable not in combined_data.columns:
    print(f"[ERROR] Target variable '{target_variable}' not found in dataset!")
    exit()

# Convert target to binary (1 indicating someone is depressed 0 indicates minimal to no depression)
depression_mapping = {
    "yes": 1, "always": 1, "often": 1, "almost always": 1, 
    "no": 0, "never": 0, "rarely": 0, "sometimes": 0
}

combined_data[target_variable] = (
    combined_data[target_variable]
    .astype(str)
    .str.lower()
    .str.strip()
    .map(depression_mapping)
)

combined_data.dropna(subset=[target_variable], inplace=True)
y = combined_data[target_variable].astype(int)

#  we are doing Feature Engineering

# Dropping timestamp if exists
if "timestamp" in combined_data.columns:
    combined_data.drop(columns=["timestamp"], inplace=True)

# Cleaning the age column
if 'age' in combined_data.columns:
    combined_data.loc[:, 'age'] = pd.to_numeric(combined_data['age'], errors='coerce')
    combined_data['age'] = combined_data['age'].fillna(combined_data['age'].median())
    combined_data.loc[:, 'age'] = combined_data['age'].astype(int)

# Filling missing values using median to obtain accurate predictions
combined_data.fillna(combined_data.median(numeric_only=True), inplace=True)

for col in combined_data.select_dtypes(include=['object']).columns:
    if col != target_variable:
        combined_data[col] = LabelEncoder().fit_transform(combined_data[col].astype(str))

# Defining features and the target variable
X = combined_data.drop(columns=[target_variable])

#training test splitting and scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# function handling  class imbalance with SMOTE
if len(np.unique(y_train)) > 1:
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
else:
    X_train_resampled, y_train_resampled = X_train_scaled, y_train

# doing model training

# 1. Random Forest Model
rf_model = RandomForestClassifier(random_state=42, class_weight="balanced")
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_search = RandomizedSearchCV(
    rf_model, rf_params, n_iter=10, cv=3, scoring='accuracy', n_jobs=-1, random_state=42
)
rf_search.fit(X_train_resampled, y_train_resampled)
best_rf = rf_search.best_estimator_

# 2. Gradient Boosting Model
gb_model = GradientBoostingClassifier(random_state=42)
gb_params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

gb_search = RandomizedSearchCV(
    gb_model, gb_params, n_iter=10, cv=3, scoring='accuracy', n_jobs=-1, random_state=42
)
gb_search.fit(X_train_resampled, y_train_resampled)
best_gb = gb_search.best_estimator_

# 3. Support Vector Machine (SVM) Model
svm_model = SVC(random_state=42, probability=True)
svm_params = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

svm_search = RandomizedSearchCV(
    svm_model, svm_params, n_iter=10, cv=3, scoring='accuracy', n_jobs=-1, random_state=42
)
svm_search.fit(X_train_resampled, y_train_resampled)
best_svm = svm_search.best_estimator_

# 4. Logistic Regression {Fully Optimized} Model
lr_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=10000, warm_start=True)
lr_params = {
    'C': np.logspace(-4, 2, 20),
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear'],
    'tol': [1e-5]
}

lr_search = RandomizedSearchCV(
    lr_model, 
    lr_params, 
    n_iter=15,
    cv=3,
    scoring='accuracy', 
    n_jobs=-1, 
    random_state=42
)
lr_search.fit(X_train_resampled, y_train_resampled)
best_lr = lr_search.best_estimator_


# Model Evaluation

models = {
    "Random Forest": best_rf,
    "Gradient Boosting": best_gb,
    "SVM": best_svm,
    "Logistic Regression": best_lr
}

print("\n[MSG INFO] Model Performance Comparisons:")
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))

#Handling interpretability of the models and visualization
# This function plots the top 10 most important features  using a horizontal bar chart.  
# It extracts feature importance values, sorts them and labels the y-axis with corresponding feature names.  
# The plot is saved as a PNG file. 

def plot_feature_importance(model, feature_names, title):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:] 
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, title):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Depressed', 'Depressed'],
                yticklabels=['Not Depressed', 'Depressed'])
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}_CM.png")
    plt.close()

def plot_roc_curve(model, X_test, y_test, title):
    """Plot ROC curve"""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}_ROC.png")
    plt.close()
#generates a SHAP  summary plot, which helps interpret the contribution of features 
def plot_shap_summary(model, X_train, feature_names, title):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    plt.figure()
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}_SHAP.png")
    plt.close()

# Creating visualizations directory
os.makedirs("visualizations", exist_ok=True)

# Generate visualization for the best model 
best_model_name = max(models.keys(), key=lambda x: accuracy_score(y_test, models[x].predict(X_test_scaled)))
best_model = models[best_model_name]

# Feature Importance Plot
if hasattr(best_model, 'feature_importances_'):
    plot_feature_importance(best_model, X.columns, f"{best_model_name} Mental Health Feature Importance")

# Confusion Matrix
y_pred_best = best_model.predict(X_test_scaled)
plot_confusion_matrix(y_test, y_pred_best, f"{best_model_name} Mental Health Confusion Matrix")

# ROC Curve
if hasattr(best_model, 'predict_proba'):
    plot_roc_curve(best_model, X_test_scaled, y_test, f"{best_model_name} Mental Health ROC Curve")

# SHAP Summary Plot 
if best_model_name in ["Random Forest", "Gradient Boosting"]:
    plot_shap_summary(best_model, X_train_scaled, X.columns, f"{best_model_name} Mental Health SHAP Summary")

# Feature Importance (from the Random Forest)
feature_importance = pd.Series(best_rf.feature_importances_, index=X.columns)
top_features = feature_importance.nlargest(10).sort_values(ascending=False)
print("\n[INFO] Top 10 Features Contributing to Prediction (Random Forest):\n", top_features)

# Save the best model (based on their accuracies)
joblib.dump(best_model, 'best_mental_health_model.joblib')
print(f"\n[SUCCESS] Best model ({best_model_name}) saved as 'best_mental_health_model deployed.joblib'")

# Save feature importance
top_features.to_csv("feature_importance.csv", index=True)
print("\n[SUCCESS] Feature importance saved as 'Mental Health feature_importance.csv'")

# Save the scaler
joblib.dump(scaler, 'scaler.joblib')
print("\n[SUCCESS] Scaler saved as 'scaler.joblib'")

print("\n[SUCCESS] All visualizations saved in 'visualizations' directory")