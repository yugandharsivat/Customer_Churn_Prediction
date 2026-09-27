# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to churn based on customer information, services, contract details, and billing information.

## Project Overview

Customer churn is an important problem for subscription-based businesses. This project uses the IBM Telco Customer Churn dataset to analyze customer behavior and build machine learning models that predict whether a customer is likely to leave the service.

The project compares two supervised classification algorithms:

- Logistic Regression
- Random Forest Classifier

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

## Dataset

The dataset contains information about 7,043 telecom customers and 21 columns.

The dataset includes information related to:

- Customer demographics
- Tenure
- Phone services
- Internet services
- Online security
- Online backup
- Device protection
- Technical support
- Streaming services
- Contract type
- Payment method
- Monthly charges
- Total charges
- Customer churn

### Target Variable

The target variable is `Churn`.

- `No` = Customer did not churn
- `Yes` = Customer churned

### Dataset Distribution

- Total Customers: 7,043
- Customers who stayed: 5,174
- Customers who churned: 1,869

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Machine Learning Workflow

The project follows these steps:

1. Load the dataset
2. Explore the dataset
3. Check data types
4. Check missing values
5. Check duplicate records
6. Convert `TotalCharges` to numeric format
7. Remove unnecessary `customerID`
8. Encode the target variable
9. Separate features and target
10. Identify numerical and categorical features
11. Handle missing values
12. Encode categorical features using One-Hot Encoding
13. Scale numerical features
14. Split the dataset into training and testing sets
15. Train Logistic Regression model
16. Train Random Forest model
17. Generate predictions
18. Evaluate model performance
19. Generate confusion matrices
20. Generate ROC curve
21. Analyze important features
22. Save the trained model

## Data Preprocessing

The following preprocessing techniques are used:

- `customerID` is removed because it does not provide useful predictive information.
- `TotalCharges` is converted from string to numeric format.
- Missing numerical values are handled using median imputation.
- Missing categorical values are handled using the most frequent value.
- Categorical features are converted into numerical features using One-Hot Encoding.
- Numerical features are standardized using StandardScaler.
- The dataset is divided into training and testing sets using an 80:20 split.

## Machine Learning Models

### Logistic Regression

Logistic Regression is used as a supervised classification algorithm to predict whether a customer will churn.

### Random Forest

Random Forest is an ensemble classification algorithm that combines multiple decision trees to make predictions.

## Model Results

The models were evaluated on the test dataset.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.38% | 64.85% | 57.22% | 60.80% | 83.59% |
| Random Forest | 78.96% | 63.45% | 49.20% | 55.42% | 81.40% |

## Feature Analysis

The project also analyzes the features that have a stronger influence on customer churn predictions using Logistic Regression coefficients.

Some of the important features identified include:

- Customer tenure
- Contract type
- Total charges
- Internet service
- Monthly charges
- Paperless billing
- Multiple lines
- Online security
- Technical support
- Streaming services

## Evaluation Metrics

### Accuracy

Measures the percentage of total predictions that are correct.

### Precision

Measures how many customers predicted as churn customers actually churned.

### Recall

Measures how many of the actual churn customers were correctly identified.

### F1 Score

Provides a balance between Precision and Recall.

### ROC-AUC

Measures the model's ability to distinguish between customers who churn and customers who do not churn.

## Visualizations

The project generates the following visualizations:

- Logistic Regression Confusion Matrix
- Random Forest Confusion Matrix
- ROC Curve
- Top Features Influencing Customer Churn

## Output

After running the project, the trained machine learning model is saved as:

```text
customer_churn_model.pkl
```

The program also displays:

- Model performance metrics
- Classification reports
- Confusion matrices
- ROC curve
- Important feature analysis
- Model comparison

## Project Structure

```text
Customer_Churn_Prediction/
│
├── Customer_Churn_Prediction.py
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── customer_churn_model.pkl
├── README.md
└── requirements.txt
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Customer_Churn_Prediction.git
```

### 2. Open the Project Folder

```bash
cd Customer_Churn_Prediction
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

## Running the Project

Run the Python script:

```bash
python Customer_Churn_Prediction.py
```

The program will:

- Load the dataset
- Clean and preprocess the data
- Train the machine learning models
- Generate predictions
- Evaluate the models
- Display visualizations
- Analyze important features
- Save the trained model

## Requirements

The project requires the following Python libraries:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

These dependencies are also listed in `requirements.txt`.

## Applications

Customer churn prediction can help businesses:

- Identify customers who may leave
- Understand customer behavior
- Analyze factors related to churn
- Support customer retention strategies
- Improve customer service decisions

## Future Improvements

Possible future improvements include:

- Hyperparameter tuning
- Handling class imbalance
- Trying additional machine learning algorithms
- Building an interactive prediction interface
- Creating a web application using Flask or Streamlit
- Adding real-time customer churn prediction
- Deploying the model as an API

## Skills Demonstrated

- Python Programming
- Machine Learning
- Supervised Classification
- Data Preprocessing
- Exploratory Data Analysis
- Feature Engineering
- Model Training
- Model Evaluation
- Data Visualization
- Scikit-learn
- Git & GitHub

## Internship Project

This project was developed as part of a Python and Machine Learning internship project portfolio.

## Author

**YUGANDHAR SIVA**

B.Tech CSE - Artificial Intelligence and Machine Learning

## Project Status

Completed