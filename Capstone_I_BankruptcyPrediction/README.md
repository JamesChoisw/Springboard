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
- Handled missing values:
  - Used KNN imputation for numerical features.
  - Applied median imputation for categorical features.
- Detected outliers using z-scores and interquartile range but retained them for analysis.
- Resolved class imbalance:
  - Applied SMOTE (Synthetic Minority Oversampling Technique) to increase the minority class proportion from 3%.
  - Resulting dataset: ~9,000 observations.

### 2. **Feature Engineering**
- Created new features by combining financial ratios (e.g., stability and profitability indicators).
- Generated dummy variables for numerical features to indicate outliers.
- Selected significant features using LASSO regression:
  - Optimal alpha: 0.0001.
  - Reduced features from 112 to 58.

### 3. **Modeling**
- Machine learning models implemented:
  - **Logistic Regression**: Optimized `C` and `max_iter`.
  - **Random Forest**: Balanced class weights and tuned hyperparameters.
  - **Support Vector Machine (SVM)**: Kernel selection and hyperparameter tuning.
  - **LightGBM**: Boosting techniques for imbalanced data.
- Evaluated models using metrics like F2/F3 to prioritize recall.

### 4. **Evaluation**
- Focused on recall over precision due to the high cost of failing to predict bankruptcy.
- Best results:
  - **Recall**: 0.82
  - **Precision**: 0.12 (with F3.16 scoring for imbalance prioritization).
- Highlighted trade-offs between recall and precision to optimize business decisions.

## Key Results
- Models demonstrated strong predictive capabilities for bankruptcy detection, emphasizing recall to minimize false negatives.
- Feature engineering and SMOTE significantly improved model performance.
- Logistic Regression and LightGBM showed competitive results, balancing interpretability and predictive power.

## Technologies Used
- **Programming**: Python
- **Libraries**:
  - Data Preprocessing: pandas, numpy, scikit-learn
  - Feature Selection: statsmodels, LASSO
  - Modeling: scikit-learn, LightGBM
  - Evaluation: metrics (F2/F3), confusion matrix, ROC-AUC
- **Visualization**: matplotlib, seaborn

## Challenges and Learnings
- Addressing severe class imbalance required creative approaches like SMOTE and F-score prioritization.
- Balancing recall and precision required tuning specific to business needs.
- Feature selection with LASSO revealed the importance of dimensionality reduction for improving model performance.

## Repository Structure
```plaintext
Capstone_I_BankruptcyPrediction/
├── data/                # Raw and processed datasets
├── notebooks/           # Jupyter notebooks for EDA, modeling, and evaluation
├── models/              # Saved model files
├── reports/             # Analysis reports and visualizations
├── README.md            # Project overview
