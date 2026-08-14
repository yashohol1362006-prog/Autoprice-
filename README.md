# 🚗 AutoPrice

AutoPrice is a machine learning project that predicts the price of a used car based on its details.

I built this project to practice the complete machine learning workflow — starting from exploring and cleaning a real dataset, preparing the data for a model, training a Linear Regression model, evaluating its performance, and finally putting the model into a simple Streamlit web application.

## What does it do?

You enter details about a car such as:

- Year
- Mileage
- Tax
- MPG
- Engine Size
- Transmission
- Fuel Type
- Make
- Model

The application then uses the trained machine learning model to estimate the car's price.

## 🛠️ Technologies I Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Linear Regression
- Streamlit
- Joblib

## 📊 Dataset

The dataset contains **72,435 car records** with 10 original features.

The main columns are:

`model`, `year`, `price`, `transmission`, `mileage`, `fuelType`, `tax`, `mpg`, `engineSize`, and `Make`.

I performed exploratory data analysis to understand the dataset, checked for missing values and duplicate rows, examined categorical features, and created visualizations to understand relationships between different features and car prices.

## 🤖 Machine Learning

For the first version of the project, I used **Linear Regression**.

Before training the model, categorical features were converted into numerical features using one-hot encoding.

I then split the data into training and testing sets and evaluated the model using:

- R² Score
- Adjusted R²
- MAE
- MSE
- RMSE

### Model Results

| Metric | Result |
|---|---:|
| R² Score | 0.867 |
| Adjusted R² | 0.865 |
| MAE | ₹2,109.52 |
| RMSE | ₹3,437.80 |

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Exploration
   ↓
Data Cleaning
   ↓
Categorical Encoding
   ↓
Train/Test Split
   ↓
Linear Regression
   ↓
Model Evaluation
   ↓
Save Trained Model
   ↓
Streamlit Application
   ↓
Car Price Prediction
