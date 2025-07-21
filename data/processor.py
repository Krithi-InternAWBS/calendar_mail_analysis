"""
Data Processor Module
Handles data transformation and preprocessing for meeting-email correlation analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import re

from logger import get_logger, LoggerMixin, log_performance
from config import DashboardConfig


class DataProcessor(LoggerMixin):
    """
    Handles data processing and transformation for the dashboard
    """
    
    def __init__(self):
        """Initialize the data processor"""
        self.config = DashboardConfig()
        self.logger.info("DataProcessor initialized")
    
    @log_performance
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main data processing pipeline
        
        Args:
            df: Raw DataFrame from Excel loader
            
        Returns:
            pd.DataFrame: Processed DataFrame ready for analysis
        """
        self.log_method_entry("process_data", shape=df.shape)
        
        try:
            # Create copy to avoid modifying original
            processed_df = df.copy()
            
            # Step 1: Clean and standardize column names
            processed_df = self._clean_column_names(processed_df)
            
            # Step 2: Parse and standardize datetime columns
            processed_df = self._process_datetime_columns(processed_df)
            
            # Step 3: Clean and standardize text columns
            processed_df = self._clean_text_columns(processed_df)
            
            # Step 4: Extract additional features
            processed_df = self._extract_features(processed_df)
            
            # Step 5: Validate processed data
            processed_df = self._validate_processed_data(processed_df)
            
            self.logger.info("Data processing completed successfully. Final shape: %s", processed_df.shape)
            self.log_method_exit("process_data", processed_df)
            
            return processed_df
            
        except Exception as e:
            error_msg = f"Error in data processing: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg) from e
    
    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize column names
        
        Args:
            df: DataFrame with potentially messy column names
            
        Returns:
            pd.DataFrame: DataFrame with cleaned column names
        """
        self.log_method_entry("_clean_column_names")
        
        # Create mapping for column name standardization
        column_mapping = {}
        
        for col in df.columns:
            # Remove extra whitespace and standardize
            clean_col = str(col).strip()
            
            # Map to expected column names if close match
            for expected_col in self.config.REQUIRED_COLUMNS:
                if self._is_similar_column(clean_col, expected_col):
                    column_mapping[col] = expected_col
                    break
            else:
                # Keep original if no match found
                column_mapping[col] = clean_col
        
        # Apply column mapping
        df_renamed = df.rename(columns=column_mapping)
        
        self.logger.info("Column name cleaning completed. Renamed %d columns", len(column_mapping))
        self.log_method_exit("_clean_column_names", df_renamed)
        
        return df_renamed
    
    def _is_similar_column(self, actual: str, expected: str) -> bool:
        """
        Check if column names are similar (case-insensitive, ignore punctuation)
        
        Args:
            actual: Actual column name
            expected: Expected column name
            
        Returns:
            bool: True if columns are similar
        """
        # Normalize both strings for comparison
        actual_norm = re.sub(r'[^\w\s]', '', actual.lower().strip())
        expected_norm = re.sub(r'[^\w\s]', '', expected.lower().strip())
        
        return actual_norm == expected_norm
    
    def _process_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse and standardize datetime columns
        
        Args:
            df: DataFrame with datetime columns
            
        Returns:
            pd.DataFrame: DataFrame with processed datetime columns
        """
        self.log_method_entry("_process_datetime_columns")
        
        datetime_columns = ['Email Time (BST/GMT)', 'Meeting Time (BST/GMT)']
        
        for col in datetime_columns:
            if col in df.columns:
                df[col] = self._parse_datetime_column(df[col], col)
        
        # Recalculate time delta if both datetime columns exist
        if all(col in df.columns for col in datetime_columns):
            df = self._recalculate_time_delta(df)
        
        self.logger.info("Datetime processing completed")
        self.log_method_exit("_process_datetime_columns", df)
        
        return df
    
    def _parse_datetime_column(self, series: pd.Series, column_name: str) -> pd.Series:
        """
        Parse datetime column with multiple format attempts
        
        Args:
            series: Pandas series with datetime strings
            column_name: Name of the column for logging
            
        Returns:
            pd.Series: Parsed datetime series
        """
        self.logger.debug("Parsing datetime column: %s", column_name)
        
        # Try each datetime format
        for fmt in self.config.DATETIME_FORMATS:
            try:
                parsed_series = pd.to_datetime(series, format=fmt, errors='coerce')
                success_rate = parsed_series.notna().mean()
                
                if success_rate > 0.8:  # If at least 80% parsed successfully
                    self.logger.info("Successfully parsed %s with format %s (%.2f%% success)", 
                                   column_name, fmt, success_rate * 100)
                    return parsed_series
                    
            except Exception as e:
                self.logger.debug("Format %s failed for column %s: %s", fmt, column_name, str(e))
                continue
        
        # If no format worked, try pandas automatic parsing
        try:
            parsed_series = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
            success_rate = parsed_series.notna().mean()
            
            self.logger.warning("Using automatic parsing for %s (%.2f%% success)", 
                              column_name, success_rate * 100)
            return parsed_series
            
        except Exception as e:
            self.logger.error("Failed to parse datetime column %s: %s", column_name, str(e))
            return series
    
    def _recalculate_time_delta(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Recalculate time delta between email and meeting times
        
        Args:
            df: DataFrame with datetime columns
            
        Returns:
            pd.DataFrame: DataFrame with updated time delta
        """
        self.log_method_entry("_recalculate_time_delta")
        
        try:
            email_time = df['Email Time (BST/GMT)']
            meeting_time = df['Meeting Time (BST/GMT)']
            
            # Calculate time difference in hours
            time_diff = (email_time - meeting_time).dt.total_seconds() / 3600
            
            df['Time Delta (hrs)'] = time_diff
            
            # Log statistics about time deltas
            valid_deltas = time_diff.dropna()
            if len(valid_deltas) > 0:
                self.logger.info("Time delta statistics - Min: %.2f hrs, Max: %.2f hrs, Mean: %.2f hrs",
                               valid_deltas.min(), valid_deltas.max(), valid_deltas.mean())
            
            self.log_method_exit("_recalculate_time_delta", df)
            return df
            
        except Exception as e:
            self.logger.warning("Failed to recalculate time delta: %s", str(e))
            return df
    
    def _clean_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize text columns
        
        Args:
            df: DataFrame with text columns
            
        Returns:
            pd.DataFrame: DataFrame with cleaned text columns
        """
        self.log_method_entry("_clean_text_columns")
        
        text_columns = [
            'Meeting Subject', 'Email Subject', 'Email Body Content',
            'Mail Sender', 'Mail Receiver', 'Organizer', 'Attendees'
        ]
        
        for col in text_columns:
            if col in df.columns:
                df[col] = self._clean_text_series(df[col], col)
        
        self.logger.info("Text cleaning completed")
        self.log_method_exit("_clean_text_columns", df)
        
        return df
    
    def _clean_text_series(self, series: pd.Series, column_name: str) -> pd.Series:
        """
        Clean individual text series
        
        Args:
            series: Text series to clean
            column_name: Column name for logging
            
        Returns:
            pd.Series: Cleaned text series
        """
        # Convert to string and handle nulls
        cleaned = series.astype(str).replace('nan', '')
        
        # Remove extra whitespace
        cleaned = cleaned.str.strip()
        
        # Remove empty strings (replace with NaN)
        cleaned = cleaned.replace('', pd.NA)
        
        self.logger.debug("Cleaned text column: %s", column_name)
        
        return cleaned
    
    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract additional features from the data
        
        Args:
            df: DataFrame to extract features from
            
        Returns:
            pd.DataFrame: DataFrame with additional features
        """
        self.log_method_entry("_extract_features")
        
        # Extract email domains
        df = self._extract_email_domains(df)
        
        # Extract time-based features
        df = self._extract_time_features(df)
        
        # Calculate content quality scores
        df = self._calculate_content_scores(df)
        
        # Extract meeting size
        df = self._extract_meeting_size(df)
        
        self.logger.info("Feature extraction completed")
        self.log_method_exit("_extract_features", df)
        
        return df
    
    def _extract_email_domains(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract email domains from sender and receiver columns
        
        Args:
            df: DataFrame with email columns
            
        Returns:
            pd.DataFrame: DataFrame with domain columns
        """
        email_columns = ['Mail Sender', 'Mail Receiver']
        
        for col in email_columns:
            if col in df.columns:
                domain_col = f"{col.replace('Mail ', '')} Domain"
                df[domain_col] = df[col].astype(str).apply(self._extract_domain)
        
        return df
    
    def _extract_domain(self, email: str) -> str:
        """
        Extract domain from email address
        
        Args:
            email: Email address string
            
        Returns:
            str: Domain or 'unknown'
        """
        try:
            if pd.isna(email) or email == 'nan' or '@' not in str(email):
                return 'unknown'
            
            domain = str(email).split('@')[-1].lower().strip()
            return domain if domain else 'unknown'
            
        except Exception:
            return 'unknown'
    
    def _extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract time-based features from datetime columns
        
        Args:
            df: DataFrame with datetime columns
            
        Returns:
            pd.DataFrame: DataFrame with time features
        """
        datetime_columns = ['Email Time (BST/GMT)', 'Meeting Time (BST/GMT)']
        
        for col in datetime_columns:
            if col in df.columns and df[col].dtype == 'datetime64[ns]':
                base_name = col.replace(' (BST/GMT)', '').replace(' ', '_').lower()
                
                # Extract time components
                df[f"{base_name}_hour"] = df[col].dt.hour
                df[f"{base_name}_day_of_week"] = df[col].dt.dayofweek
                df[f"{base_name}_month"] = df[col].dt.month
                df[f"{base_name}_quarter"] = df[col].dt.quarter
        
        return df
    
    def _calculate_content_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate content quality scores based on keywords
        
        Args:
            df: DataFrame with text content
            
        Returns:
            pd.DataFrame: DataFrame with content scores
        """
        text_columns = ['Email Subject', 'Email Body Content']
        
        for col in text_columns:
            if col in df.columns:
                score_col = f"{col.replace(' ', '_').lower()}_score"
                df[score_col] = df[col].apply(self._calculate_keyword_score)
        
        return df
    
    def _calculate_keyword_score(self, text: str) -> int:
        """
        Calculate keyword-based quality score for text
        
        Args:
            text: Text to analyze
            
        Returns:
            int: Quality score
        """
        if pd.isna(text) or text == 'nan':
            return 0
        
        text_lower = str(text).lower()
        total_score = 0
        
        # Score based on keyword categories
        weights = self.config.get_keyword_score_weights()
        
        for category, keywords in self.config.ENGAGEMENT_KEYWORDS.items():
            category_score = sum(1 for keyword in keywords if keyword in text_lower)
            total_score += category_score * weights.get(category, 1)
        
        return total_score
    
    def _extract_meeting_size(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract meeting size from attendees column
        
        Args:
            df: DataFrame with attendees column
            
        Returns:
            pd.DataFrame: DataFrame with meeting size
        """
        if 'Attendees' in df.columns:
            df['Meeting_Size'] = df['Attendees'].apply(self._count_attendees)
        
        return df
    
    def _count_attendees(self, attendees_text: str) -> int:
        """
        Count number of attendees from attendees string
        
        Args:
            attendees_text: String containing attendee information
            
        Returns:
            int: Number of attendees
        """
        if pd.isna(attendees_text) or attendees_text == 'nan':
            return 0
        
        # Split by common delimiters and count
        delimiters = [';', ',', '\n', '|']
        attendees_str = str(attendees_text)
        
        for delimiter in delimiters:
            if delimiter in attendees_str:
                return len([x.strip() for x in attendees_str.split(delimiter) if x.strip()])
        
        # If no delimiters found, assume single attendee
        return 1 if attendees_str.strip() else 0
    
    def _validate_processed_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate processed data quality
        
        Args:
            df: Processed DataFrame
            
        Returns:
            pd.DataFrame: Validated DataFrame
        """
        self.log_method_entry("_validate_processed_data", shape=df.shape)
        
        # Check for extreme time deltas
        if 'Time Delta (hrs)' in df.columns:
            max_delta = self.config.VALIDATION_RULES['max_time_delta']
            extreme_deltas = df['Time Delta (hrs)'].abs() > max_delta
            
            if extreme_deltas.any():
                extreme_count = extreme_deltas.sum()
                self.logger.warning("Found %d records with extreme time deltas (>%d hrs)", 
                                  extreme_count, max_delta)
        
        # Log final data quality metrics
        self._log_data_quality_metrics(df)
        
        self.log_method_exit("_validate_processed_data", df)
        return df
    
    def _log_data_quality_metrics(self, df: pd.DataFrame) -> None:
        """
        Log data quality metrics for the processed DataFrame
        
        Args:
            df: DataFrame to analyze
        """
        total_records = len(df)
        
        # Calculate null rates for key columns
        key_columns = ['Meeting Subject', 'Email Subject', 'Meeting Time (BST/GMT)', 'Email Time (BST/GMT)']
        
        for col in key_columns:
            if col in df.columns:
                null_rate = df[col].isnull().mean()
                self.logger.info("Column '%s' null rate: %.2f%%", col, null_rate * 100)
        
        # Log unique value counts
        if 'Meeting Subject' in df.columns:
            unique_meetings = df['Meeting Subject'].nunique()
            self.logger.info("Unique meetings: %d (%.2f%% of total)", 
                           unique_meetings, (unique_meetings / total_records) * 100)
        
        if 'Email Subject' in df.columns:
            unique_emails = df['Email Subject'].nunique()
            self.logger.info("Unique emails: %d (%.2f%% of total)", 
                           unique_emails, (unique_emails / total_records) * 100)