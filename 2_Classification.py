
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🤖 Classification Models")

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
            fpr, tpr, _ = roc_curve(y_test, probs[:,1])
            ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)
