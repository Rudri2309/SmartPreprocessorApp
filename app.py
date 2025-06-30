import streamlit as st
import pandas as pd
import os
import sqlite3
import json
from sqlalchemy import create_engine
from preprocessor import SmartPreprocessor
import io

st.set_page_config(page_title="SmartPreprocessor", layout="wide")
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

if df is not None:
    st.write("📋 Data Preview:")
    st.dataframe(df.head())

    if st.button("🔧 Run SmartPreprocessor"):
        tool = SmartPreprocessor(df)

        original_shape = df.shape
        original_columns = df.columns.tolist()

        tool.drop_empty_columns()
        tool.convert_dates()
        tool.clean_phones()
        tool.validate_emails()
        tool.validate_websites()
        tool.validate_zip_codes()
        tool.check_negative_values()
        tool.detect_outliers_iqr()
        tool.clean_text_columns()
        tool.drop_duplicates()

        cleaned_df = tool.get_cleaned_data()
        summary = tool.get_summary()

        st.subheader("📝 Professional Cleaning Summary")
        st.json(summary)

        # Allow user to download the cleaned file
        export_format = st.selectbox("Choose download format", ["CSV", "Excel", "JSON"])
        if export_format == "CSV":
            output = io.StringIO()
            cleaned_df.to_csv(output, index=False)
            st.download_button("Download Cleaned CSV", data=output.getvalue(),
                               file_name="Cleaned_Output.csv", mime="text/csv")
        elif export_format == "Excel":
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                cleaned_df.to_excel(writer, index=False)
            st.download_button("Download Cleaned Excel", data=output.getvalue(),
                               file_name="Cleaned_Output.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        elif export_format == "JSON":
            output = io.StringIO()
            cleaned_df.to_json(output, orient="records", lines=True)
            st.download_button("Download Cleaned JSON", data=output.getvalue(),
                               file_name="Cleaned_Output.json", mime="application/json")

        # Download summary report as JSON
        st.download_button("📄 Download Summary Report", data=json.dumps(summary, indent=2),
                           file_name="Cleaning_Summary.json", mime="application/json")
