import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Predictive Maintenance", layout="wide")
st.title("⚙️ Predictive Maintenance — IoT Machine Failure Prediction")

uploaded_file = st.file_uploader("📂 Upload your IoT sensor CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Sample Data")
    st.write(df.head())

    st.info("🧠 Loading trained model...")
    if os.path.exists("models/rf_model.joblib"):
        model = joblib.load("models/rf_model.joblib")

        if "failure" in df.columns:
            X = df.drop(columns=["failure"])
        else:
            X = df.copy()

        preds = model.predict(X)
        df["Predicted_Failure"] = preds
        st.success("✅ Prediction Completed!")
        st.write(df.head())
        st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv")
    else:
        st.warning("Trained model not found. Please upload the model file into /models folder.")
else:
    st.info("👆 Upload a CSV file to begin.")
