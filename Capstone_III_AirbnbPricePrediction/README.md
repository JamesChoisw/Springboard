# Capstone III: Airbnb Price Prediction

This project is part of my Springboard Data Science Bootcamp and focuses on predicting Airbnb listing prices using various features such as location, amenities, and property characteristics. The objective is to build an accurate regression model that can help hosts and travelers understand the price determinants of Airbnb properties.

## Objective
Predict Airbnb listing prices using a machine learning regression approach by analyzing multiple features such as geographical location, property amenities, and host details.

## Dataset
- **Source**: Publicly available Airbnb dataset with features on property characteristics, host information, and pricing.
- **Features**:
  - **Numerical Features**: Bedrooms, bathrooms, beds, etc.
  - **Categorical Features**: Neighborhood, zipcode, property type, amenities, etc.
  - **Textual Features**: Property descriptions, reviews, and titles.
- **Target**: The `Log Price` variable representing the nightly listing price.

## Methodology
### 1. **Data Preprocessing**
- **Handling Missing Values**:
  - Imputed missing values in numerical features (e.g., bedrooms, bathrooms) using KNN imputation.
  - Imputed categorical features (e.g., neighborhood, zip code) based on geographical proximity (longitude and latitude) using Random Forest.
- **Outlier Analysis**:
  - Checked for outliers using the interquartile range but retained them due to realistic values.
- **Feature Encoding**:
  - Applied one-hot encoding and binary encoding for categorical features.

### 2. **Feature Engineering**
- **Geospatial Features**:
  - Calculated the distance of each property from downtown using latitude and longitude coordinates.
- **Text Features**:
  - Applied TF-IDF vectorization on textual data (e.g., property descriptions).
- **New Features**:
  - Created numerical features to capture listing characteristics and property uniqueness.
  - Grouping ammenities using K-Means Clustering

### 3. **Modeling**
- **Machine Learning Models Implemented**:
  - Lasso Regression
  - Random Forest Regression
  - Gradient Boosting Machines (GBM)
  - XGBoost
  - MLP (ANN)
  - Weighted Average Ensemble: Optimized the Weights
- **Model Tuning**:
  - Applied hyperparameter tuning using Grid Search and cross-validation.

### 4. **Evaluation**
- **Performance Metrics**:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R-squared (R²) for model goodness of fit.
- **Training and Test Comparison**:
  - Ensured no significant performance difference between training and test sets, indicating robust model performance.

## Key Results
- The model achieved strong predictive performance, with Weighted Averaged Ensemble and XGBoost Machines showing competitive results.
- Room Type is the most critical feature, significantly influencing a listing's value and appeal.
- Features like Bedrooms and Accommodates, describing property size and capacity, are highly impactful.
- Location-related attributes (e.g., distance to downtown, city, zipcode) strongly affect desirability.
- Amenities and property description provide meaningful explanatory power for listing quality.


## Technologies Used
- **Programming**: Python
- **Libraries**:
  - Data Preprocessing: pandas, numpy, scikit-learn
  - Feature Engineering: scikit-learn, category_encoders
  - Modeling: scikit-learn, XGBoost
- **Visualization**: matplotlib, seaborn

---

**Contact**: Suk Won Choi  
[Email: jameschoisw00@gmail.com] | [[LinkedIn Profile](https://www.linkedin.com/in/james-sukwon-choi/)]
