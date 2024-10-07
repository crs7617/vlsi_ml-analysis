import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit title and file uploader
st.title("ML Analysis for Individual CSV File")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Step 1: Read the uploaded CSV
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.dataframe(df.head())
    
    # Step 2: Allow the user to select the target column
    target_col = st.selectbox('Select the target column for prediction:', df.columns)
    
    # Ensure at least 2 columns exist for features and target
    if len(df.columns) > 1:
        # Step 3: Preprocess the data
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        # Step 4: Train a Random Forest Regressor Model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        reg = RandomForestRegressor()
        reg.fit(X_train, y_train)

        # Step 5: Run predictions
        y_pred = reg.predict(X_test)

        # Step 6: Display regression metrics
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
        st.write(f"R-squared: {r2_score(y_test, y_pred)}")

        # Step 7: Plot feature importance
        feature_importance = pd.Series(reg.feature_importances_, index=X.columns).sort_values(ascending=False)
        st.write("### Feature Importance")
        fig, ax = plt.subplots()
        sns.barplot(x=feature_importance, y=feature_importance.index, ax=ax)
        st.pyplot(fig)

        # Step 8: Plot predicted vs actual values
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
