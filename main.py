import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


st.title("ML Analysis for Individual CSV File")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.dataframe(df.head())
    
    target_col = st.selectbox('Select the target column for prediction:', df.columns)
    
    if len(df.columns) > 1:

        X = df.drop(target_col, axis=1)
        y = df[target_col]


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        reg = RandomForestRegressor()
        reg.fit(X_train, y_train)

        y_pred = reg.predict(X_test)

        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
        st.write(f"R-squared: {r2_score(y_test, y_pred)}")


        feature_importance = pd.Series(reg.feature_importances_, index=X.columns).sort_values(ascending=False)
        st.write("### Feature Importance")
        fig, ax = plt.subplots()
        sns.barplot(x=feature_importance, y=feature_importance.index, ax=ax)
        st.pyplot(fig)
        st.write("### Predicted vs Actual")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        st.pyplot(fig)
        
    else:
        st.write("Error: The dataset must have at least two columns (features + target).")
else:
    st.write("Please upload a CSV file to analyze.")
