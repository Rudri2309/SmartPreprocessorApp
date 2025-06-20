# -*- coding: utf-8 -*-
"""App.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1DNk5FoN4w-PLFeqqjEApTyDb7zp10DKb
"""

import streamlit as st
import pandas as pd
import os
import sqlite3
import json
from sqlalchemy import create_engine
from preprocessor import SmartPreprocessor

st.set_page_config(page_title="SmartPreprocessor Tool", layout="wide")
st.title("SmartPreprocessor")

source_type = st.selectbox("Select your data source type",
                           ["CSV", "Excel", "JSON", "SQLite (.db)", "PostgreSQL"])

df = None
filename = None

if source_type in ["CSV", "Excel", "JSON", "SQLite (.db)"]:
    uploaded_file = st.file_uploader(f"Upload your {source_type} file")

    if uploaded_file:
        filename = uploaded_file.name
        ext = os.path.splitext(filename)[1]

        try:
            if source_type == "CSV":
                df = pd.read_csv(uploaded_file)
            elif source_type == "Excel":
                df = pd.read_excel(uploaded_file)
            elif source_type == "JSON":
                df = pd.read_json(uploaded_file)
            elif source_type == "SQLite (.db)":
                conn = sqlite3.connect(uploaded_file)
                table_names = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
                table_to_load = st.selectbox("Select table", table_names["name"])
                df = pd.read_sql(f"SELECT * FROM {table_to_load}", conn)
                conn.close()
            st.success("✅ Data loaded successfully.")
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
else:
    st.subheader("PostgreSQL Connection")
    pg_host = st.text_input("Host", value="localhost")
    pg_port = st.text_input("Port", value="5432")
    pg_db = st.text_input("Database name")
    pg_user = st.text_input("Username")
    pg_password = st.text_input("Password", type="password")
    pg_table = st.text_input("Table name")

    if st.button("Connect and Load"):
        try:
            engine = create_engine(f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}")
            df = pd.read_sql(f"SELECT * FROM {pg_table}", con=engine)
            st.success("✅ Connected and data loaded.")
        except Exception as e:
            st.error(f"❌ Connection failed: {e}")

# 🔄 Preprocessing
if df is not None:
    st.write("📋 Data Preview:")
    st.dataframe(df.head())

    if st.button("🔧 Run SmartPreprocessor"):
        tool = SmartPreprocessor(df)

        # Track original stats
        original_shape = df.shape
        original_columns = df.columns.tolist()

        # Apply transformations
        tool.drop_empty_columns()
        tool.convert_dates()
        tool.clean_phones()
        tool.validate_emails()
        tool.validate_websites()
        tool.clean_text_columns()
        tool.drop_duplicates()

        cleaned_df = tool.get_cleaned_data()

        # 📊 Summary Report
        summary = {
            "Original Rows": original_shape[0],
            "Original Columns": original_shape[1],
            "Columns Removed (Empty >90%)": list(set(original_columns) - set(cleaned_df.columns)),
            "Remaining Rows": cleaned_df.shape[0],
            "Remaining Columns": cleaned_df.shape[1],
            "Added Columns": [col for col in cleaned_df.columns if col.startswith("Valid ")],
            "Duplicate Rows Dropped": original_shape[0] - cleaned_df.shape[0]
        }

        st.subheader("📝 Cleaning Summary Report")
        st.json(summary)

        # ⬇️ Export Options
        export_format = st.selectbox("Choose download format", ["CSV", "Excel", "JSON"])
        if export_format == "CSV":
            export_filename = "Cleaned_Output.csv"
            cleaned_df.to_csv(export_filename, index=False)
        elif export_format == "Excel":
            export_filename = "Cleaned_Output.xlsx"
            cleaned_df.to_excel(export_filename, index=False)
        elif export_format == "JSON":
            export_filename = "Cleaned_Output.json"
            cleaned_df.to_json(export_filename, orient="records", lines=True)

        with open(export_filename, "rb") as f:
            st.download_button(f"📥 Download Cleaned Data ({export_format})", f, file_name=export_filename)
