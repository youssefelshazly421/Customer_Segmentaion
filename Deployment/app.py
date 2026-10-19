import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

# Load the trained model and scaler
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
df = pd.read_csv("Mall_Customers.csv")

# Cluster descriptions based on our training results
cluster_names = {
    0: "Average Income - Average Spending",
    1: "High Income - High Spending",
    2: "Low Income - High Spending",
    3: "High Income - Low Spending",
    4: "Low Income - Low Spending"
}

# Main title
st.title("🛍️ Customer Segmentation")
st.write(
    "Discover customer groups using "
    "Machine Learning and K-Means clustering."
)

st.divider()

# Sidebar inputs
st.sidebar.header("Customer Information")

income = st.sidebar.slider(
    "Annual Income (k$)",
    min_value=15,
    max_value=137,
    value=60
)

spending = st.sidebar.slider(
    "Spending Score (1-100)",
    min_value=1,
    max_value=100,
    value=50
)

# Predict button
if st.sidebar.button("Predict", type="primary"):

    # Prepare the input in the same feature order used during training
    customer = pd.DataFrame(
        [[income, spending]],
        columns=["Annual Income (k$)", "Spending Score (1-100)"]
    )

    # Apply the saved scaler
    customer_scaled = scaler.transform(customer)

    # Predict the nearest K-Means cluster
    cluster = int(kmeans.predict(customer_scaled)[0])

    st.subheader("Prediction Result")

    st.success(f"Customer belongs to Cluster {cluster}")

    st.write(f"**Customer Segment:** {cluster_names[cluster]}")

    # Plot existing customers and the new customer
    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        df["Annual Income (k$)"],
        df["Spending Score (1-100)"],
        c=kmeans.labels_,
        cmap="Set2",
        s=55,
        alpha=0.7
    )

    # Convert the saved centroids to the original scale
    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="X",
        s=220,
        color="black",
        label="Cluster Centers"
    )

    # Highlight the new customer
    ax.scatter(
        income,
        spending,
        marker="*",
        s=350,
        color="red",
        edgecolor="black",
        label="New Customer"
    )

    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.set_title("Customer Segmentation Results")
    ax.legend()
    ax.grid(alpha=0.2)

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "The predicted cluster is based on the trained K-Means model. "
        "Cluster descriptions summarize the average patterns observed "
        "in the training dataset."
    )

else:
    st.info(
        "Adjust the customer information in the sidebar, "
        "then click Predict to see the result."
    )