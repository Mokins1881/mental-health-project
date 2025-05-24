
# Mental Health Prediction Project

![Machine Learning](https://img.shields.io/badge/-Machine%20Learning-blueviolet)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Library-Scikit--learn-orange)
![HealthTech](https://img.shields.io/badge/Domain-HealthTech-success)

A machine learning project that predicts depression likelihood based on student mental health survey data, using multiple classification algorithms with explainable AI techniques.

## 📌 Project Overview

This project analyzes mental health survey data from multiple sources to:
- Predict depression likelihood using machine learning
- Identify key contributing factors
- Provide model interpretability using SHAP and feature importance
- Compare performance of 4 different classification algorithms

## 📊 Key Results

| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Gradient Boosting    | 89.6%    | 0.88      | 0.81   | 0.84     |
| Random Forest        | 88.3%    | 0.87      | 0.77   | 0.82     |
| SVM                  | 80.5%    | 0.82      | 0.54   | 0.65     |
| Logistic Regression  | 76.6%    | 0.68      | 0.58   | 0.62     |

**Top 5 Predictive Features**:
1. Depression type (12.4% importance)
2. Depression score (10.5%)
3. Marital status (6.3%)
4. Depression severity (5.7%)
5. Age (4.5%)

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Mokins1881/mental-health-project.git
   cd mental-health-project
Install requirements:

bash
pip install -r requirements.txt
🚀 Usage
Run the main script:

bash
python mentalhealth.py
This will:

Load and preprocess the data

Train and evaluate 4 ML models

Generate visualizations in /visualizations

Save the best model (best_mental_health_model.joblib)

📂 File Structure
mental-health-project/
├── data/
│   ├── pone.0304132.s001.csv          # Primary dataset 1
│   ├── cleaned_pone.0304132.s002.csv  # Primary dataset 2
│   └── Student Mental health.csv      # Secondary dataset
├── visualizations/                    # Generated plots
├── mentalhealth.py                    # Main Python script
├── best_mental_health_model.joblib    # Saved model
├── scaler.joblib                      # Feature scaler
└── README.md                          # Project documentation
📈 Sample Visualizations
Feature Importance
ROC Curve

🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.


📧 Contact
For questions or collaborations, please contact [Maxwel Muriuki] at [muriukimwoki@gmail.com]

