import pandas as pd
import numpy as np
import re
import validators
import phonenumbers

class SmartPreprocessor:
    def __init__(self, df):
        self.df = df.copy()
        self.phone_cols = [col for col in self.df.columns if 'phone' in col.lower()]
        self.email_cols = [col for col in self.df.columns if 'email' in col.lower()]
        self.date_cols = [col for col in self.df.columns if 'date' in col.lower()]
        self.website_cols = [col for col in self.df.columns if 'website' in col.lower()]
        self.zip_cols = [col for col in self.df.columns if 'zip' in col.lower()]
        self.text_cols = self._detect_text_fields()
        self.numeric_cols = self.df.select_dtypes(include='number').columns.tolist()

        self.summary = {
            "original_shape": self.df.shape,
            "columns_removed": [],
            "duplicate_rows_dropped": 0,
            "validations_added": [],
            "outliers_flagged": {}
        }

    def _detect_text_fields(self):
        likely_texts = []
        for col in self.df.columns:
            if self.df[col].dtype == 'object' and any(x in col.lower() for x in ['name', 'city', 'country', 'company']):
                likely_texts.append(col)
        return likely_texts

    def drop_empty_columns(self, threshold=0.9):
        null_frac = self.df.isnull().mean()
        cols_to_drop = null_frac[null_frac > threshold].index.tolist()
        self.df.drop(columns=cols_to_drop, inplace=True)
        self.summary["columns_removed"] = cols_to_drop

    def convert_dates(self):
        for col in self.date_cols:
            self.df[col] = pd.to_datetime(self.df[col], errors='coerce')

    def clean_phones(self):
        def clean_number(val):
            try:
                parsed = phonenumbers.parse(str(val), None)
                if phonenumbers.is_possible_number(parsed) and phonenumbers.is_valid_number(parsed):
                    return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
                else:
                    return np.nan
            except Exception:
                return np.nan

        for col in self.phone_cols:
            self.df[col] = self.df[col].apply(clean_number)

    def validate_emails(self):
        for col in self.email_cols:
            validation_col = f"Valid {col}"
            self.df[validation_col] = self.df[col].apply(lambda x: validators.email(str(x).strip()))
            invalid_count = (~self.df[validation_col]).sum()
            self.summary[f"Invalid Emails in {col}"] = int(invalid_count)
            self.summary["validations_added"].append(validation_col)

    def validate_websites(self):
        for col in self.website_cols:
            validation_col = f"Valid {col}"
            self.df[validation_col] = self.df[col].apply(lambda x: validators.url(str(x).strip()))
            invalid_count = (~self.df[validation_col]).sum()
            self.summary[f"Invalid URLs in {col}"] = int(invalid_count)
            self.summary["validations_added"].append(validation_col)

    def validate_zip_codes(self):
        for col in self.zip_cols:
            validation_col = f"Valid {col}"
            self.df[validation_col] = self.df[col].astype(str).str.match(r'^\d{5}$')
            invalid_count = (~self.df[validation_col]).sum()
            self.summary[f"Invalid ZIP Codes in {col}"] = int(invalid_count)
            self.summary["validations_added"].append(validation_col)

    def check_negative_values(self):
        for col in self.numeric_cols:
            invalid_count = len(self.df[self.df[col] < 0])
            if invalid_count > 0:
                self.summary[f"Negative Values in {col}"] = int(invalid_count)

    def detect_outliers_iqr(self):
        for col in self.numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = len(self.df[(self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))])
            if outlier_count > 0:
                self.summary["outliers_flagged"][col] = int(outlier_count)

    def clean_text_columns(self):
        for col in self.text_cols:
            self.df[col] = self.df[col].apply(lambda x: x.strip().title() if isinstance(x, str) else x)

    def drop_duplicates(self, subset=None):
        before = self.df.shape[0]
        if subset and subset in self.df.columns:
            self.df.drop_duplicates(subset=subset, inplace=True)
        else:
            self.df.drop_duplicates(inplace=True)
        after = self.df.shape[0]
        self.summary["duplicate_rows_dropped"] = before - after

    def get_cleaned_data(self):
        return self.df

    def get_summary(self):
        self.summary["final_shape"] = self.df.shape
        return self.summary