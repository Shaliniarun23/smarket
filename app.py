import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Consumer Insights Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Title
st.title("📊 Consumer Insights Dashboard")
st.markdown("""
Welcome to the **Consumer Insights Dashboard**.  
Use the left sidebar to navigate through the following analytical modules:
- 📊 Data Visualization  
- 🤖 Classification  
- 📌 Clustering  
- 🔗 Association Rule Mining  
- 📈 Regression Analysis  
""")
