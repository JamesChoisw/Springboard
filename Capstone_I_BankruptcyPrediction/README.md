# Capstone I: Bankruptcy Prediction

This project is part of my Springboard Data Science Bootcamp, focusing on predicting corporate bankruptcy using machine learning techniques and financial ratios. The objective is to develop a reliable model that identifies companies at risk of bankruptcy to aid decision-making for stakeholders.

## Objective
Predict the likelihood of bankruptcy for companies based on their financial metrics, aiming to assist investors, creditors, and policymakers in making informed decisions.

## Dataset
- **Source**: The dataset includes financial ratios and indicators for various companies.  
- **Features**: 
  - Financial ratios related to stability, profitability, growth, and activity.
  - A single categorical feature: `Liability-Assets_Flag`.
  - Total features after preprocessing: 112.
- **Target**: Binary variable indicating bankruptcy (`1` for bankrupt, `0` for non-bankrupt).

## Methodology
### 1. **Data Preprocessing**
- Detected outliers using z-scores and interquartile range but retained them for analysis.
- Resolved class imbalance:
  - Applied SMOTE (Synthetic Minority Oversampling Technique) to increase the minority class proportion from 3%.

### 2. **Feature Engineering**
- Created new features by combining financial ratios (e.g., stability and profitability indicators).
- Generated dummy variables for numerical features to indicate outliers.
- Selected significant features using LASSO regression & Random Forest:

### 3. **Modeling**
- Machine learning models implemented:
  - **Logistic Regression**
  - **Random Forest**
  - **LightGBM**
  - **XGBoost**
  - **Balanced Random Forest**
- Tuned Hyperparameters Using Grid Search
- Evaluated models using metrics like F-beta score and recall to reflect the Asymmetry in Costs between Type I and Type II Errors

### 4. **Evaluation**
- Feature selection using Random Forest outperformed Lasso Regression in terms of model performance
- Focused on recall over precision due to the high cost of failing to predict bankruptcy.
- Best results: Balanced Random Forest Model
  - **Recall**: 0.82
  - **Precision**: 0.12 (with F3.16 scoring for imbalance prioritization)
  - **F-beta score**: 0.72
- Highlighted trade-offs between recall and precision to optimize business decisions.

## Key Results
- The best model is the Balanced Random Forest
- Feature engineering significantly improved model performance.
- Logistic Regression showed competitive results, balancing interpretability and predictive power.

## Technologies Used
- **Programming**: Python
- **Libraries**:
  - Data Preprocessing: pandas, numpy, scikit-learn
  - Feature Selection: statsmodels, LASSO
  - Modeling: scikit-learn, LightGBM, XGBoost
- **Visualization**: matplotlib, seaborn
