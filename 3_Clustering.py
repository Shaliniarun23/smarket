
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📌 Clustering Analysis")

uploaded_file = st.file_uploader("Upload dataset for clustering", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df_numeric = df.select_dtypes(include=['int64', 'float64'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numeric)

    st.write("### Elbow Method")
    wcss = []
    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    fig, ax = plt.subplots()
    ax.plot(range(2, 11), wcss, marker='o')
    ax.set_title("Elbow Curve")
    ax.set_xlabel("No of Clusters")
    ax.set_ylabel("WCSS")
    st.pyplot(fig)

    n_clusters = st.slider("Select number of clusters", 2, 10, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    st.write("### Clustered Data", df.head())

    st.download_button("Download Clustered Data", df.to_csv(index=False), file_name="clustered_data.csv")
