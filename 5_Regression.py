
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.title("📈 Regression Analysis")

uploaded_file = st.file_uploader("Upload CSV for regression", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    target_col = st.selectbox("Select Target Column", df.columns)
    X = pd.get_dummies(df.drop(columns=[target_col]))
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Decision Tree": DecisionTreeRegressor()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "RMSE": mean_squared_error(y_test, preds, squared=False),
            "R2": r2_score(y_test, preds)
        }

    st.write("### Regression Results")
    st.dataframe(pd.DataFrame(results).T.round(3))
