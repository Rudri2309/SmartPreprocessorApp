import pandas as pd
import numpy as np
import re
import validators
import phonenumbers

class SmartPreprocessor:
    def __init__(self, df):
        self.df = df.copy()

        self.phone_cols = self._match_cols(['phone', 'contact', 'mobile', 'cell'])
        self.email_cols = self._match_cols(['email', 'e-mail'])
        self.date_cols = self._match_cols(['date', 'dob', 'admission', 'discharge'])
        self.website_cols = self._match_cols(['website', 'web', 'url', 'link'])
        self.zip_cols = self._match_cols(['zip', 'zipcode', 'postal'])
        self.id_cols = self._match_cols(['id', 'patient', 'record', 'case'])
        self.numeric_cols = self.df.select_dtypes(include='number').columns.tolist()
        self.text_cols = self._match_cols(['name', 'city', 'country', 'state', 'company', 'clinic', 'doctor', 'hospital'])

        self.summary = {
            "original_shape": self.df.shape,
            "columns_removed": [],
            "duplicate_rows_dropped": 0,
            "validations_added": [],
            "outliers_flagged": {},
            "nested_fields_flagged": []
        }

        # Check for nested JSON fields
        for col in self.df.columns:
            if self.df[col].apply(lambda x: isinstance(x, dict) or isinstance(x, list)).any():
                self.summary["nested_fields_flagged"].append(col)

    def _match_cols(self, keywords):
        return [col for col in self.df.columns if any(k in col.lower() for k in keywords)]

    def drop_empty_columns(self, threshold=0.9):
        null_frac = self.df.isnull().mean()
        cols_to_drop = null_frac[null_frac > threshold].index.tolist()
        self.df.drop(columns=cols_to_drop, inplace=True)
        self.summary["columns_removed"] = cols_to_drop

    def convert_dates(self):
        for col in self.date_cols:
            self.df[col] = pd.to_datetime(self.df[col], errors='coerce')

    def clean_numeric_fields(self):
        for col in self.df.columns:
            if self.df[col].dtype == 'object' and self.df[col].str.contains(r'\*', na=False).any():
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        self.numeric_cols = self.df.select_dtypes(include='number').columns.tolist()

    # Replace your clean_phones() with pattern check
    def clean_phones(self):
        for col in self.phone_cols:
            def is_invalid(val):
                try:
                    parsed = phonenumbers.parse(str(val), None)
                    return not (phonenumbers.is_possible_number(parsed) and phonenumbers.is_valid_number(parsed))
                except:
                    return True

            original_invalid = self.df[col].apply(is_invalid).sum()
            self.df[col] = self.df[col].apply(lambda x: np.nan if is_invalid(x) else phonenumbers.format_number(phonenumbers.parse(str(x), None), phonenumbers.PhoneNumberFormat.E164) if x else np.nan)
            remaining_invalid = self.df[col].isnull().sum()

            validation_col = f"Valid {col}"
            self.summary[f"Original Invalid Phones in {col}"] = int(original_invalid)
            self.summary[f"Remaining Invalid Phones in {col}"] = int(remaining_invalid)
            self.summary["validations_added"].append(validation_col)


    def validate_emails(self):
        for col in self.email_cols:
            def is_invalid(val):
                return not validators.email(str(val).strip())
            original_invalid = self.df[col].apply(is_invalid).sum()
            validation_col = f"Valid {col}"
            self.df[validation_col] = self.df[col].apply(lambda x: not is_invalid(x))
            remaining_invalid = (~self.df[validation_col]).sum()

            self.summary[f"Original Invalid Emails in {col}"] = int(original_invalid)
            self.summary[f"Remaining Invalid Emails in {col}"] = int(remaining_invalid)
            self.summary["validations_added"].append(validation_col)

    def validate_websites(self):
        for col in self.website_cols:
            original_invalid = self.df[~self.df[col].astype(str).str.startswith("http")].shape[0]
            validation_col = f"Valid {col}"
            self.df[validation_col] = self.df[col].apply(lambda x: validators.url(str(x).strip()))
            remaining_invalid = (~self.df[validation_col]).sum()
            self.summary[f"Original Invalid URLs in {col}"] = int(original_invalid)
            self.summary[f"Remaining Invalid URLs in {col}"] = int(remaining_invalid)
            self.summary["validations_added"].append(validation_col)

    def validate_zip_codes(self):
        for col in self.zip_cols:
            original_invalid = self.df[~self.df[col].astype(str).str.match(r'^\d{5}$')].shape[0]
            validation_col = f"Valid {col}"
            self.df[validation_col] = self.df[col].astype(str).str.match(r'^\d{5}$')
            remaining_invalid = (~self.df[validation_col]).sum()
            self.summary[f"Original Invalid ZIPs in {col}"] = int(original_invalid)
            self.summary[f"Remaining Invalid ZIPs in {col}"] = int(remaining_invalid)
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

    def drop_duplicates(self):
        before = self.df.shape[0]
        if self.id_cols:
            self.df.drop_duplicates(subset=self.id_cols, inplace=True)
        else:
            self.df.drop_duplicates(inplace=True)
        after = self.df.shape[0]
        self.summary["duplicate_rows_dropped"] = before - after

    def get_cleaned_data(self):
        return self.df

    def get_summary(self):
        self.summary["final_shape"] = self.df.shape

        original_rows = self.summary["original_shape"][0]
        final_rows = self.summary["final_shape"][0]
        rows_dropped = original_rows - final_rows
        pct_rows_dropped = (rows_dropped / original_rows * 100) if original_rows else 0

        self.summary["rows_dropped"] = int(rows_dropped)
        self.summary["percent_rows_dropped"] = round(pct_rows_dropped, 2)

        invalid_counts = {k: v for k, v in self.summary.items() if "Remaining Invalid" in k and v > 0}
        original_invalids = {k: v for k, v in self.summary.items() if "Original Invalid" in k}
        negative_counts = {k: v for k, v in self.summary.items() if "Negative" in k}
        outliers = self.summary.get("outliers_flagged", {})
        total_outliers = sum(outliers.values())

        health_report = []
        for k in original_invalids:
            after_k = k.replace("Original", "Remaining")
            before = self.summary[k]
            after = self.summary.get(after_k, 0)
            if before == 0 and after == 0:
                continue  # skip if no invalids at all
            pct_of_total = round(after / original_rows * 100, 2) if original_rows else 0
            improvement = round(((before - after) / before * 100), 2) if before else 0

            health_report.append({
                "Field": k.replace("Original Invalid ", ""),
                "Before": before,
                "After": after,
                "% of Total Rows": pct_of_total,
                "% Improvement": improvement
            })

        report = {
            "Data Shape": {
                "Original Rows": original_rows,
                "Final Rows":