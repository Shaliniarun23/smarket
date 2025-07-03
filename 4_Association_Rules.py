
import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.title("🔗 Association Rule Mining")

uploaded_file = st.file_uploader("Upload CSV with transactional fields", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    trans_col = st.selectbox("Select Transactional Column", df.columns)
    transactions = df[trans_col].dropna().apply(lambda x: x.split(", ")).tolist()

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    min_support = st.slider("Min Support", 0.01, 1.0, 0.1)
    min_conf = st.slider("Min Confidence", 0.1, 1.0, 0.5)

    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)
    top_rules = rules.sort_values(by="confidence", ascending=False).head(10)

    st.write("### Top 10 Association Rules by Confidence")
    st.dataframe(top_rules)
