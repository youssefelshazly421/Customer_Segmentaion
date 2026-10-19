Customer Segmentation using K-Means

Project Overview

This project applies unsupervised machine learning to segment mall customers based on their annual income and spending score.

The goal is to identify customer groups with similar characteristics and support more targeted marketing strategies.

Objectives

- Explore and analyze customer data.
- Prepare features for machine learning.
- Apply K-Means clustering.
- Compare cluster counts using the Elbow Method and Silhouette Score.
- Visualize customer segments.
- Build an interactive web application using Streamlit.

Dataset

The project uses the Mall Customers dataset, containing customer information such as age, gender, annual income, and spending score.

The clustering model uses two features:

- Annual Income (k$)
- Spending Score (1–100)

Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit
- Jupyter Notebook

Machine Learning Approach

The selected features are standardized using "StandardScaler". K-Means is then trained to group customers into five clusters.

The Elbow Method and Silhouette Score are used to compare different numbers of clusters.

The final model achieved a Silhouette Score of 0.555 among the tested cluster counts.

Streamlit Application

The application allows users to enter a customer's annual income and spending score, click Predict, and view the predicted cluster along with a visualization of the customer segments.

Project Structure

- "app.py" — Streamlit application
- "customer_segmentation.ipynb" — Data analysis and model development
- "Mall_Customers.csv" — Dataset
- "kmeans_model.pkl" — Trained K-Means model
- "scaler.pkl" — Fitted feature scaler
- "requirements.txt" — Required Python packages

How to Run

1. Clone or download this repository.

2. Install the required packages:
   
   "pip install -r requirements.txt"

3. Run the application:
   
   "python -m streamlit run app.py"

Conclusion

This project demonstrates how unsupervised machine learning can uncover customer segments and support data-driven marketing decisions.
