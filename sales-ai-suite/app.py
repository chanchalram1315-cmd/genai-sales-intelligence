import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import streamlit as st
import pandas as pd
from utils import clean_data
from models import perform_customer_segmentation
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

st.set_page_config(page_title="GenAI Sales Intelligence Suite", layout="wide")

st.title("📊 GenAI-Powered Sales & Retail Intelligence Suite")
st.write("Upload your sales dataset, clean it instantly, and chat with your data using AI.")

st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
        
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())
    
    cleaned_df = clean_data(df)
    
    st.subheader("Cleaned & Standardized Data Preview")
    st.dataframe(cleaned_df.head())
    
    st.divider()
    st.subheader("💬 Chat with your Data (GenAI Agent)")
    
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to start chatting with the dataset.")
    else:
        llm = OpenAI(api_token=api_key)
        sdf = SmartDataframe(cleaned_df, config={"llm": llm})
        
        query = st.text_input("Ask a question about your data (e.g., 'What are the top 3 products by sales?'):")
        
        if query:
            with st.spinner("AI is analyzing your data..."):
                try:
                    response = sdf.chat(query)
                    st.success("Result:")
                    st.write(response)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    
    st.divider()
    st.subheader("🤖 Predictive Analytics & Customer Segmentation")
    if st.button("Run K-Means Clustering"):
        segmented_data, message = perform_customer_segmentation(cleaned_df)
        st.info(message)
        if segmented_data is not None:
            st.dataframe(segmented_data.head(10))
            st.scatter_chart(segmented_data, x=segmented_data.columns[0], y=segmented_data.columns[1], color="cluster")
else:
    st.info("Please upload a CSV or Excel file to get started.")