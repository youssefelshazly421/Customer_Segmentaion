import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

# Load trained model, scaler, and dataset
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
df = pd.read_csv("Mall_Customers.csv")

# Cluster descriptions based on K-Means training results
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
    "Discover customer groups using machine learning. "
    "Choose K-Means or DBSCAN from the sidebar."
)
st.divider()

# Sidebar controls
st.sidebar.header("Customer Information")

algorithm = st.sidebar.selectbox(
    "Clustering Algorithm",
    ["K-Means", "DBSCAN"]
)

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

if algorithm == "DBSCAN":
    st.sidebar.caption(
        "DBSCAN settings must match the model trained and saved from the notebook."
    )

if st.sidebar.button("Predict", type="primary"):
    features = ["Annual Income (k$)", "Spending Score (1-100)"]
    customer = pd.DataFrame([[income, spending]], columns=features)
    customer_scaled = scaler.transform(customer)

    fig, ax = plt.subplots(figsize=(10, 6))

    if algorithm == "K-Means":
        cluster = int(kmeans.predict(customer_scaled)[0])
        plot_labels = kmeans.labels_

        st.subheader("Prediction Result")
        st.success(f"Customer belongs to Cluster {cluster}")
        st.write(f"**Customer Segment:** {cluster_names.get(cluster, 'Cluster ' + str(cluster))}")

        ax.scatter(
            df[features[0]], df[features[1]],
            c=plot_labels, cmap="Set2", s=55, alpha=0.7
        )

        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        ax.scatter(
            centers[:, 0], centers[:, 1],
            marker="X", s=220, color="black",
            label="Cluster Centers"
        )

    else:
        try:
            dbscan = joblib.load("dbscan_model.pkl")
        except FileNotFoundError:
            plt.close(fig)
            st.error(
                "Could not find dbscan_model.pkl. Train DBSCAN in the notebook "
                "and save it first (instructions below the code)."
            )
            st.stop()

        plot_labels = dbscan.labels_

        # DBSCAN has no native predict() method. Assign the new customer to
        # the nearest core sample only if it is within DBSCAN's eps distance.
        core_samples = dbscan.components_
        core_labels = dbscan.labels_[dbscan.core_sample_indices_]

        st.subheader("DBSCAN Result")

        if len(core_samples) == 0:
            cluster = -1
        else:
            distances = np.linalg.norm(core_samples - customer_scaled[0], axis=1)
            nearest_idx = int(np.argmin(distances))
            if distances[nearest_idx] <= dbscan.eps:
                cluster = int(core_labels[nearest_idx])
            else:
                cluster = -1

        if cluster == -1:
            st.warning(
                "DBSCAN classifies this customer as Noise / not assigned to a cluster."
            )
        else:
            st.success(f"Customer is assigned to DBSCAN Cluster {cluster}")

        # Plot existing customer clusters
        ax.scatter(
            df[features[0]], df[features[1]],
            c=plot_labels, cmap="Set1", s=55, alpha=0.7
        )

        if len(core_samples) > 0:
            core_original = scaler.inverse_transform(core_samples)
            ax.scatter(
                core_original[:, 0], core_original[:, 1],
                marker="x", s=45, color="black", label="DBSCAN Core Samples"
            )

    # Highlight the new customer
    ax.scatter(
        income, spending,
        marker="*", s=350, color="red",
        edgecolor="black", label="New Customer"
    )

    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.set_title(f"Customer Segmentation Results - {algorithm}")
    ax.legend()
    ax.grid(alpha=0.2)

    st.pyplot(fig)
    plt.close(fig)

    if algorithm == "DBSCAN":
        st.caption(
            "DBSCAN does not provide a native predict() method. The new customer "
            "is assigned to the nearest core sample if it is within eps; otherwise "
            "it is shown as Noise. This is a practical approximation, not an official "
            "DBSCAN prediction API."
        )
    else:
        st.caption(
            "The K-Means cluster prediction uses the trained model. Cluster "
            "descriptions summarize patterns observed in the training dataset."
        )
else:
    st.info(
        "Adjust the customer information in the sidebar, choose an algorithm, "
        "then click Predict."
    )
