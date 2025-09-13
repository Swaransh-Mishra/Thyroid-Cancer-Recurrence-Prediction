🧬 Thyroid Cancer Recurrence Prediction










📖 Project Overview

Thyroid cancer recurrence is a major concern in oncology, and accurate risk prediction models can guide early intervention and better treatment planning.

This project delivers a full-stack ML solution that includes:

Data preprocessing & feature engineering

Model development, hyperparameter tuning, and evaluation

Model explainability with SHAP

Deployment as a Streamlit web application

By combining performance metrics with interpretability, this project demonstrates job-ready ML engineering skills, covering data science, MLOps, and deployment.

🎯 Key Highlights

📊 Dataset preprocessing with encoding, scaling, and cleaning

🔍 Fine-tuned Random Forest classifier using GridSearchCV

📈 Achieved 97% accuracy with ROC-AUC 0.9884

✅ Evaluation with confusion matrix, ROC curve, precision-recall, calibration curve

🧾 Explainability with SHAP feature importance plots

🌐 Deployed as a Streamlit web app for recruiters & stakeholders

📂 Project Structure
📁 Thyroid-Cancer-Recurrence-Prediction
│── app.py                           # Streamlit app
│── predictions.py                   # Preprocessing & prediction pipeline
│── tuned_random_forest_model.pkl    # Saved Random Forest model
│── scaler.pkl                       # StandardScaler object
│── thyroid_dataset.csv              # Original dataset
│── thyroid_dataset_preprocessed.csv # Cleaned dataset
│── THYROID CANCER PREDICTION.ipynb  # Jupyter Notebook (EDA & training)
│── README.md                        # Project documentation

🛠️ Tech Stack

Programming Language: Python 🐍

Machine Learning: scikit-learn (Random Forest, preprocessing, evaluation)

Data Analysis: pandas, numpy

Visualization: matplotlib, seaborn

Explainability: SHAP

Deployment: Streamlit

⚙️ Workflow
🧹 1. Data Preprocessing

Encoded categorical features with custom mappings

StandardScaler applied to numerical features

Final processed dataset ready for model training

🤖 2. Model Development

Trained multiple models, Random Forest performed best

Hyperparameter tuning using GridSearchCV

Final tuned model saved as tuned_random_forest_model.pkl

📊 3. Model Evaluation

Best Parameters Found:

{'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_depth': None, 'bootstrap': True}


Classification Report (Tuned Random Forest):

Class	Precision	Recall	F1-Score	Support
0	0.96	1.00	0.98	55
1	1.00	0.91	0.95	22
Accuracy	0.97			77
Macro Avg	0.98	0.95	0.97	77
Weighted Avg	0.97	0.97	0.97	77

📈 ROC-AUC Score: 0.9884

🔍 Model Explainability

SHAP plots were used to analyze feature impact on predictions:

🔴 Red → Feature increases recurrence risk

🔵 Blue → Feature decreases recurrence risk

📏 Longer bars → Stronger influence

Example SHAP plot:

🚀 Running the Project Locally

Clone the repo:

git clone https://github.com/yourusername/thyroid-cancer-prediction.git
cd thyroid-cancer-prediction


Install dependencies:

pip install -r requirements.txt


Run the app:

streamlit run app.py

🌐 Live Demo

👉 Click here to try the Streamlit Web App

👤 Author

Swaransh Mishra
📧 your.email@example.com

💼 LinkedIn