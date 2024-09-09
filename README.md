# E-commerce Product Recommendation System

This project implements a **Collaborative Filtering-based Recommendation System** for an e-commerce platform. It suggests personalized product recommendations to users based on their interaction data such as clicks, ratings, and purchase history. The goal is to enhance the user experience and increase sales by showing relevant products to the users.

## Features

- **Data Preprocessing**: Cleans missing data and handles categorical conversion.
- **Exploratory Data Analysis (EDA)**: Provides insights into user interactions, popular products, category distribution, and ratings.
- **Collaborative Filtering (SVD)**: Uses matrix factorization to predict user-product interactions.
- **Personalized Recommendations**: Provides the top N recommended products for each user.
- **Model Evaluation**: Evaluates the recommendation model using RMSE and MAE.

---

## Table of Contents

1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Code Workflow](#code-workflow)
   - Data Preprocessing
   - Exploratory Data Analysis
   - Building the Recommendation Model
   - Generating Recommendations
4. [Usage](#usage)
5. [Evaluation](#evaluation)
6. [Future Improvements](#future-improvements)

---

## Installation

### Prerequisites

Make sure you have Python installed (version 3.x). You will also need the following Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn surprise
```

### Steps to Clone and Run the Project

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/HacktiveMindset/E-commerce-Product-Recommendation-System.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ecommerce-recommendation-system
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Python file to generate recommendations:
   ```bash
   python recommendation_system.py
   ```

---

## Dataset

The dataset is assumed to be customer interaction data, containing the following columns:

- `user_id`: Unique identifier for the user.
- `product_id`: Unique identifier for the product.
- `product_name`: Name of the product.
- `product_category`: The category to which the product belongs.
- `ratings`: User rating for the product (scale of 1-5).
- `user_clicks`: Number of times the user clicked on the product.
- `purchase_history`: Whether the product was purchased by the user (`Yes` or `No`).

> **Note:** Replace the dataset path in the code with your own dataset path if different.

---

## Code Workflow

### 1. Data Preprocessing

The first step is to clean and prepare the data. We convert categorical variables like `purchase_history` to numerical values and handle missing data using the forward fill method:

```python
# Convert 'purchase_history' from categorical to numerical (Yes -> 1, No -> 0)
data['purchase_history'] = data['purchase_history'].map({'Yes': 1, 'No': 0})

# Handle missing values using forward fill
data.ffill(inplace=True)
```

### 2. Exploratory Data Analysis (EDA)

We provide visual insights into popular products, product category distribution, and ratings distribution:

```python
# Popular products based on user clicks
top_products = data.groupby('product_name')['user_clicks'].sum().sort_values(ascending=False).head(10)
```

The code uses `seaborn` and `matplotlib` to create plots:

- **Top 10 most clicked products** (bar plot)
- **Product category distribution** (count plot)
- **Ratings distribution** (histogram)

### 3. Building the Recommendation Model

We use **Collaborative Filtering** (Matrix Factorization with Singular Value Decomposition - SVD) to predict missing product ratings for users. This allows us to recommend products they are likely to rate highly:

```python
from surprise import SVD, Dataset, Reader
reader = Reader(rating_scale=(1, 5))
surprise_data = Dataset.load_from_df(recommendation_data, reader)

# Train-test split and model training
trainset = surprise_data.build_full_trainset()
model = SVD()
model.fit(trainset)
```

### 4. Generating Recommendations

The system generates personalized recommendations for a user based on predicted product ratings. We calculate predictions for products the user hasn't rated and return the top 10 recommended products:

```python
# Predict ratings for all products the user has not rated
product_ids = data['product_id'].unique()
recommendable_products = [pid for pid in product_ids if pid not in user_rated_products]
predictions = [model.predict(user_id, product_id).est for product_id in recommendable_products]

# Get top 10 recommendations
recommended_products = pd.DataFrame({
    'product_id': recommendable_products,
    'predicted_rating': predictions
}).sort_values(by='predicted_rating', ascending=False).head(10)
```

---

## Usage

1. **Run the Code**:
   - Execute `recommendation_system.py` to start the recommendation engine.
2. **Generate Recommendations**:
   - The system will generate personalized recommendations for users and output them in the console.
3. **Change User**:
   - You can modify the `user_id` variable in the code to get recommendations for any specific user.

---

## Evaluation

We evaluate the model's performance using **cross-validation** techniques such as **RMSE (Root Mean Squared Error)** and **MAE (Mean Absolute Error)**:

```python
from surprise.model_selection import cross_validate
cross_validate(model, surprise_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
```

These metrics help determine how well the model predicts the ratings, guiding potential improvements.

---

## Future Improvements

- **Hybrid Recommender**: Combine collaborative filtering with content-based methods to improve recommendation accuracy.
- **Deep Learning**: Integrate neural collaborative filtering models for better feature extraction and prediction accuracy.
- **User Behavior Analysis**: Incorporate more user behavior data (e.g., browsing time, session patterns) for more personalized recommendations.

---

## Conclusion

This project builds a **Collaborative Filtering-based Recommendation System** to enhance user shopping experience. It leverages user behavior data and machine learning to suggest personalized products, increasing engagement and sales on the platform.

Feel free to improve or modify the system as needed!

---
