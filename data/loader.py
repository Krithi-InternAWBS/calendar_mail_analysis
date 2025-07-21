"""
Excel Data Loader Module
Handles loading and initial validation of Excel files for meeting-email correlation analysis
"""

import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Union, Optional, List
import io
import re

from logger import get_logger, LoggerMixin, log_performance
from config import DashboardConfig


class ExcelDataLoader(LoggerMixin):
    """
    Handles Excel file loading and initial validation for the dashboard
    """
    
    def __init__(self):
        """Initialize the Excel data loader"""
        self.config = DashboardConfig()
        self.logger.info("ExcelDataLoader initialized")
    
    @log_performance
    def load_from_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from Excel file path with dynamic header detection
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        self.log_method_entry("load_from_file", file_path=str(file_path))
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            error_msg = f"Excel file not found: {file_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            self.logger.info("Loading Excel file: %s", file_path)
            
            # Find the header row dynamically
            header_row = self._find_header_row(file_path)
            self.logger.info("Found headers at row: %d", header_row)
            
            # Load the Excel file with correct header row
            df = pd.read_excel(file_path, engine='openpyxl', header=header_row)
            
            self.logger.info("Successfully loaded Excel file with shape: %s", df.shape)
            
            # Clean column names
            df.columns = df.columns.astype(str).str.strip()
            
            # Validate the loaded data
            validated_df = self._validate_data(df, str(file_path))
            
            self.log_method_exit("load_from_file", validated_df)
            return validated_df
            
        except Exception as e:
            error_msg = f"Error loading Excel file {file_path}: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg) from e
    
    @log_performance
    def load_from_upload(self, uploaded_file) -> pd.DataFrame:
        """
        Load data from Streamlit uploaded file with dynamic header detection
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            ValueError: If file format is invalid
        """
        self.log_method_entry("load_from_upload", filename=uploaded_file.name)
        
        try:
            self.logger.info("Loading uploaded file: %s", uploaded_file.name)
            
            # Read file content
            file_content = uploaded_file.read()
            
            # Create BytesIO object for pandas
            file_buffer = io.BytesIO(file_content)
            
            # Find header row dynamically
            header_row = self._find_header_row_from_buffer(file_buffer)
            self.logger.info("Found headers at row: %d", header_row)
            
            # Reset buffer position
            file_buffer.seek(0)
            
            # Load Excel data with correct header row
            df = pd.read_excel(file_buffer, engine='openpyxl', header=header_row)
            
            self.logger.info("Successfully loaded uploaded file with shape: %s", df.shape)
            
            # Clean column names
            df.columns = df.columns.astype(str).str.strip()
            
            # Validate the loaded data
            validated_df = self._validate_data(df, uploaded_file.name)
            
            self.log_method_exit("load_from_upload", validated_df)
            return validated_df
            
        except Exception as e:
            error_msg = f"Error loading uploaded file {uploaded_file.name}: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg) from e
    
    def _validate_data(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """
        Validate loaded Excel data against expected schema
        
        Args:
            df: DataFrame to validate
            source_name: Source file name for logging
            
        Returns:
            pd.DataFrame: Validated DataFrame
            
        Raises:
            ValueError: If validation fails
        """
        self.log_method_entry("_validate_data", shape=df.shape, source=source_name)
        
        validation_errors = []
        
        # Check if DataFrame is empty
        if df.empty:
            validation_errors.append("Excel file is empty")
        
        # Check minimum record count
        if len(df) < self.config.VALIDATION_RULES['min_records']:
            validation_errors.append(
                f"Insufficient records: {len(df)} < {self.config.VALIDATION_RULES['min_records']}"
            )
        
        # Check for required columns
        missing_columns = self._check_required_columns(df)
        if missing_columns:
            validation_errors.append(f"Missing required columns: {missing_columns}")
        
        # Check data quality
        quality_issues = self._check_data_quality(df)
        if quality_issues:
            validation_errors.extend(quality_issues)
        
        # Report validation results
        if validation_errors:
            error_msg = f"Data validation failed for {source_name}: {'; '.join(validation_errors)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.info("Data validation passed for %s", source_name)
        self.log_method_exit("_validate_data", df)
        return df
    
    def _find_header_row(self, file_path: Path) -> int:
        """
        Dynamically find the row containing column headers
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            int: Row number containing headers (0-indexed)
        """
        self.log_method_entry("_find_header_row", file_path=str(file_path))
        
        try:
            # Read first 20 rows to find headers
            df_sample = pd.read_excel(file_path, engine='openpyxl', header=None, nrows=20)
            
            # Look for required columns in each row
            for row_idx in range(len(df_sample)):
                row_values = df_sample.iloc[row_idx].astype(str).str.strip().str.lower()
                
                # Check how many required columns we can find in this row
                matches = 0
                for required_col in self.config.REQUIRED_COLUMNS:
                    required_col_lower = required_col.lower().strip()
                    
                    # Check for exact matches or partial matches
                    for cell_value in row_values:
                        if (required_col_lower in cell_value or 
                            self._is_similar_column(cell_value, required_col_lower)):
                            matches += 1
                            break
                
                # If we find at least 60% of required columns, this is likely the header row
                match_ratio = matches / len(self.config.REQUIRED_COLUMNS)
                self.logger.debug("Row %d: Found %d/%d matches (%.1f%%)", 
                                row_idx, matches, len(self.config.REQUIRED_COLUMNS), match_ratio * 100)
                
                if match_ratio >= 0.6:  # At least 60% of columns found
                    self.logger.info("Found header row at index %d with %.1f%% match", 
                                   row_idx, match_ratio * 100)
                    return row_idx
            
            # If no good match found, default to row 0
            self.logger.warning("No clear header row found, defaulting to row 0")
            return 0
            
        except Exception as e:
            self.logger.error("Error finding header row: %s", str(e))
            return 0
    
    def _find_header_row_from_buffer(self, file_buffer: io.BytesIO) -> int:
        """
        Find header row from uploaded file buffer
        
        Args:
            file_buffer: BytesIO buffer containing Excel data
            
        Returns:
            int: Row number containing headers (0-indexed)
        """
        self.log_method_entry("_find_header_row_from_buffer")
        
        try:
            # Save current position
            current_pos = file_buffer.tell()
            
            # Read sample data
            df_sample = pd.read_excel(file_buffer, engine='openpyxl', header=None, nrows=20)
            
            # Reset buffer position
            file_buffer.seek(current_pos)
            
            # Use same logic as file-based detection
            for row_idx in range(len(df_sample)):
                row_values = df_sample.iloc[row_idx].astype(str).str.strip().str.lower()
                
                matches = 0
                for required_col in self.config.REQUIRED_COLUMNS:
                    required_col_lower = required_col.lower().strip()
                    
                    for cell_value in row_values:
                        if (required_col_lower in cell_value or 
                            self._is_similar_column(cell_value, required_col_lower)):
                            matches += 1
                            break
                
                match_ratio = matches / len(self.config.REQUIRED_COLUMNS)
                
                if match_ratio >= 0.6:
                    self.logger.info("Found header row at index %d with %.1f%% match", 
                                   row_idx, match_ratio * 100)
                    return row_idx
            
            self.logger.warning("No clear header row found in buffer, defaulting to row 0")
            return 0
            
        except Exception as e:
            self.logger.error("Error finding header row from buffer: %s", str(e))
            return 0
    
    def _is_similar_column(self, actual: str, expected: str) -> bool:
        """
        Check if column names are similar (case-insensitive, flexible matching)
        
        Args:
            actual: Actual column name
            expected: Expected column name
            
        Returns:
            bool: True if columns are similar
        """
        # Normalize both strings for comparison
        actual_norm = re.sub(r'[^\w\s]', '', str(actual).lower().strip())
        expected_norm = re.sub(r'[^\w\s]', '', str(expected).lower().strip())
        
        # Check for exact match
        if actual_norm == expected_norm:
            return True
        
        # Check if expected is contained in actual (for partial matches)
        if expected_norm in actual_norm or actual_norm in expected_norm:
            return True
        
        # Check for key word matches (split by spaces and check overlap)
        actual_words = set(actual_norm.split())
        expected_words = set(expected_norm.split())
        
        # If there's significant word overlap, consider it a match
        if actual_words and expected_words:
            overlap = actual_words.intersection(expected_words)
            overlap_ratio = len(overlap) / max(len(actual_words), len(expected_words))
            return overlap_ratio >= 0.5  # At least 50% word overlap
        
        return False
    
    def _check_required_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Check if all required columns are present (with flexible matching)
        
        Args:
            df: DataFrame to check
            
        Returns:
            List[str]: List of missing columns
        """
        missing_columns = []
        found_columns = []
        
        # Convert actual column names to string and clean them
        actual_columns = [str(col).strip() for col in df.columns]
        
        for required_col in self.config.REQUIRED_COLUMNS:
            found = False
            
            # Look for exact or similar matches
            for actual_col in actual_columns:
                if self._is_similar_column(actual_col, required_col):
                    found = True
                    found_columns.append(f"{required_col} -> {actual_col}")
                    break
            
            if not found:
                missing_columns.append(required_col)
        
        if found_columns:
            self.logger.info("Found column mappings: %s", found_columns)
        
        if missing_columns:
            self.logger.warning("Missing columns detected: %s", missing_columns)
            # Also log what columns we actually have for debugging
            self.logger.info("Available columns in file: %s", actual_columns)
        
        return missing_columns
    
    def _check_data_quality(self, df: pd.DataFrame) -> List[str]:
        """
        Check data quality issues
        
        Args:
            df: DataFrame to check
            
        Returns:
            List[str]: List of quality issues
        """
        quality_issues = []
        
        # Check fill rates for required columns
        for col in self.config.REQUIRED_COLUMNS:
            if col in df.columns:
                fill_rate = df[col].notna().mean()
                min_fill_rate = self.config.VALIDATION_RULES['required_fill_rate']
                
                if fill_rate < min_fill_rate:
                    quality_issues.append(
                        f"Low fill rate for '{col}': {fill_rate:.2%} < {min_fill_rate:.2%}"
                    )
                    self.logger.warning("Low fill rate for column %s: %.2f%%", col, fill_rate * 100)
        
        # Check for duplicate records
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            self.logger.warning("Found %d duplicate records", duplicate_count)
        
        # Check email format if email columns exist
        email_columns = ['Mail Sender', 'Mail Receiver']
        for email_col in email_columns:
            if email_col in df.columns:
                self._validate_email_format(df, email_col)
        
        return quality_issues
    
    def _validate_email_format(self, df: pd.DataFrame, email_col: str) -> None:
        """
        Validate email format in specified column
        
        Args:
            df: DataFrame containing email column
            email_col: Name of email column to validate
        """
        email_regex = self.config.VALIDATION_RULES['email_format_regex']
        
        # Filter out null values
        email_series = df[email_col].dropna()
        
        if len(email_series) == 0:
            return
        
        # Check email format
        valid_emails = email_series.astype(str).str.match(email_regex, na=False)
        invalid_count = (~valid_emails).sum()
        
        if invalid_count > 0:
            invalid_rate = invalid_count / len(email_series)
            self.logger.warning(
                "Found %d invalid email formats in column '%s' (%.2f%%)",
                invalid_count, email_col, invalid_rate * 100
            )
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Generate summary statistics for loaded data
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            dict: Summary statistics
        """
        self.log_method_entry("get_data_summary", shape=df.shape)
        
        summary = {
            'total_records': len(df),
            'columns': list(df.columns),
            'column_count': len(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'null_counts': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }
        
        # Add specific metrics for our use case
        if 'Meeting Subject' in df.columns:
            summary['unique_meetings'] = df['Meeting Subject'].nunique()
        
        if 'Email Subject' in df.columns:
            summary['unique_emails'] = df['Email Subject'].nunique()
        
        if 'Mail Sender' in df.columns:
            summary['unique_senders'] = df['Mail Sender'].nunique()
        
        self.logger.info("Generated data summary for %d records", len(df))
        self.log_method_exit("get_data_summary", summary)
        
        return summary