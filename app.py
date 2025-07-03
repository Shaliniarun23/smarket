# Folder structure:
# consumer_dashboard/
# ├── app.py
# ├── requirements.txt
# ├── README.md
# └── pages/
#     ├── 1_Data_Visualization.py
#     ├── 2_Classification.py
#     ├── 3_Clustering.py
#     ├── 4_Association_Rules.py
#     └── 5_Regression.py

# -------------------------- app.py --------------------------
import streamlit as st

st.set_page_config(
    page_title="Consumer Insights Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("\U0001F4CA Consumer Insights Dashboard")
st.markdown("""
Welcome to the **Consumer Insights Dashboard**.  
Use the left sidebar to navigate through the following analytical modules:
- \U0001F4CA Data Visualization  
- ☰ Classification  
- \U0001F4CC Clustering  
- \U0001F4D0 Association Rule Mining  
- \U0001F4C9 Regression Analysis
""")

# -------------------------- requirements.txt --------------------------
# streamlit
# pandas
# numpy
# scikit-learn
# matplotlib
# seaborn
# mlxtend
# plotly

# -------------------------- pages/1_Data_Visualization.py --------------------------
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("\U0001F9EE Data Visualization")

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

# -------------------------- pages/2_Classification.py --------------------------
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns

st.title("☰ Classification Models")

uploaded_file = st.file_uploader("Upload dataset with labels (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    target_col = st.selectbox("Select Target Column", df.columns)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X = pd.get_dummies(X)
    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        report = classification_report(y_test, preds, output_dict=True)
        results[name] = report['weighted avg']

    st.write("### Classification Report")
    st.dataframe(pd.DataFrame(results).T)

    model_choice = st.selectbox("Select Model for Confusion Matrix", list(models.keys()))
    cm = confusion_matrix(y_test, models[model_choice].predict(X_test))
    st.write("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.write("### ROC Curve")
    fig, ax = plt.subplots()
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
            fpr, tpr, _ = roc_curve(y_test, probs[:, 1])
            ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

# -------------------------- pages/3_Clustering.py --------------------------
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.title("\U0001F4CC Clustering Analysis")

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

# -------------------------- pages/4_Association_Rules.py --------------------------
import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.title("\U0001F517 Association Rule Mining")

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

# -------------------------- pages/5_Regression.py --------------------------
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.title("\U0001F4C9 Regression Analysis")

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
