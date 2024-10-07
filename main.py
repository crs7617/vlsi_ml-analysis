import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Streamlit title and file uploader
st.title("ML Analysis for Individual CSV File")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Step 1: Read the uploaded CSV
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.dataframe(df.head())

    # Example: Ensure 'target' is present in your dataset, else modify accordingly
    if 'target' in df.columns:
        # Step 2: Preprocess the data
        X = df.drop('target', axis=1)
        y = df['target']

        # Step 3: Train a Random Forest Model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)

        # Step 4: Run predictions
        y_pred = clf.predict(X_test)

        # Step 5: Display accuracy score
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")

        # Example of plotting a confusion matrix or other plots
        st.write("### Prediction Distribution")
        fig, ax = plt.subplots()
        ax.bar(['Class 1', 'Class 2'], [sum(y_pred == 0), sum(y_pred == 1)])
        st.pyplot(fig)
    else:
        st.write("Error: 'target' column not found in the dataset.")
else:
    st.write("Please upload a CSV file to analyze.")
