# Rossmann Store Sales Analysis and Prediction

This project focuses on analyzing customer purchasing behavior and predicting store sales through machine learning and deep learning techniques. The tasks are divided into two primary sections:

1. **Exploration of Customer Purchasing Behavior**
2. **Prediction of Store Sales**

## Task 1: Exploration of Customer Purchasing Behavior

### 1.1 Data Cleaning
A robust data cleaning pipeline ensures reproducibility and consistency by:
- Imputing missing values using the median.
- Removing outliers in the `Sales` column using the Interquartile Range (IQR) method.

### 1.2 Exploratory Data Analysis (EDA)
Key questions drive our EDA, with insights communicated through visualizations and statistical summaries:

- **Promotions Distribution**: Comparing training and test sets to ensure consistent promotional representation.
- **Holiday Sales Behavior**: Analyzing sales variations before, during, and after holidays.
- **Seasonal Trends**: Identifying peak periods, such as Christmas and Easter.
- **Sales vs. Customer Correlation**: Investigating the relationship between customer count and sales.
- **Impact of Promotions**: Assessing promotional effectiveness on customer attraction and spending.
- **Store Openings/Closures**: Examining sales trends during store hours and after new store launches.
- **Assortment & Competitor Distance**: Exploring how product variety and competitor proximity influence sales.

## Task 2: Prediction of Store Sales

### 2.1 Preprocessing
Data is transformed into a model-ready format by:
- Extracting features from datetime columns (e.g., weekdays, days to holidays).
- Scaling data for algorithms that depend on distance metrics.

### 2.2 Sklearn Pipelines & Modeling
Using `RandomForestRegressor` within an sklearn pipeline ensures modularity and ease of deployment. 

### 2.3 Loss Function
`Mean Absolute Error (MAE)` is chosen for its interpretability and robustness against outliers.

### 2.4 Post-Prediction Analysis
Feature importance is analyzed to identify key sales drivers, and confidence intervals are estimated for prediction reliability.

### 2.5 Model Serialization
Models are serialized with timestamps to facilitate version tracking and daily predictions.

### 2.6 Deep Learning with LSTM
A Long Short-Term Memory (LSTM) model predicts future sales using time series data, preprocessed for stationarity and transformed into a supervised learning format.

## Task 3: Model Serving API

### Introduction
The trained models are deployed for real-time predictions through a REST API, enabling external systems to request and receive sales forecasts.

### Objectives
1. **Framework Selection**: Choose between Flask, FastAPI, or Django REST Framework.
2. **Model Loading**: Load serialized models for inference.
3. **API Endpoints**: Define endpoints for input data and sales predictions.
4. **Request Handling**: Preprocess inputs, run predictions, and return structured responses.
5. **Deployment**: Deploy the API on a cloud platform or web server for accessibility and scalability.

### Methodology

- **Framework Selection**: 
  - *Flask*: Lightweight and ideal for small to medium-sized APIs.
  - *FastAPI*: Modern, with interactive documentation and asynchronous support.
  - *Django REST Framework*: Suitable for larger projects with complex needs.
  
- **Model Loading**: Use `joblib` or `pickle` to deserialize the models.
  
- **API Endpoints**: 
  - `POST /predict`: Accepts JSON input and returns predictions.

- **Request Handling**: Validate and preprocess input, run model inference, and format the output as JSON.
  
- **Deployment**: Deploy to platforms like AWS, Heroku, or Google Cloud, ensuring security and scalability.

---

## Conclusion
This project provides a detailed analysis of customer purchasing behavior and a predictive framework for store sales. By integrating EDA, machine learning, and deep learning techniques, we gain actionable insights and build models capable of accurate sales forecasting.

