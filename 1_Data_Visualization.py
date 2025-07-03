
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🧮 Data Visualization")

uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())

    if st.checkbox("Show summary statistics"):
        st.write(df.describe(include='all'))

    st.write("### Average Spend by Gender")
    fig, ax = plt.subplots()
    sns.barplot(data=df, x='Gender', y='Avg Spend per Visit', ax=ax)
    st.pyplot(fig)

    st.write("### Spend Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Avg Spend per Visit'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)
