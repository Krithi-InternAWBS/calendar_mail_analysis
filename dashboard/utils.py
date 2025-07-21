"""
Dashboard Utility Functions
Helper functions for UI components, data processing, and NLP analysis
"""

import pandas as pd
import streamlit as st
import re
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import base64
import io

from logger import get_logger, LoggerMixin, log_performance
from config import DashboardConfig


class DashboardUtils(LoggerMixin):
    """
    Utility class containing helper functions for the dashboard
    """
    
    def __init__(self):
        """Initialize dashboard utilities"""
        self.config = DashboardConfig()
        self.logger.info("DashboardUtils initialized")
    
    @staticmethod
    def format_number(value: Union[int, float], format_type: str = "auto") -> str:
        """
        Format numbers for display with appropriate units
        
        Args:
            value: Number to format
            format_type: Format type ('auto', 'percentage', 'currency', 'integer')
            
        Returns:
            str: Formatted number string
        """
        if pd.isna(value):
            return "N/A"
        
        if format_type == "percentage":
            return f"{value:.1f}%"
        elif format_type == "currency":
            return f"${value:,.2f}"
        elif format_type == "integer":
            return f"{int(value):,}"
        elif format_type == "auto":
            if isinstance(value, float):
                if value < 1:
                    return f"{value:.3f}"
                elif value < 100:
                    return f"{value:.2f}"
                else:
                    return f"{value:,.1f}"
            else:
                return f"{value:,}"
        else:
            return str(value)
    
    @staticmethod
    def format_duration(hours: float) -> str:
        """
        Format duration from hours to human-readable format
        
        Args:
            hours: Duration in hours
            
        Returns:
            str: Formatted duration string
        """
        if pd.isna(hours):
            return "N/A"
        
        if hours < 1:
            minutes = int(hours * 60)
            return f"{minutes} min"
        elif hours < 24:
            return f"{hours:.1f} hrs"
        else:
            days = int(hours // 24)
            remaining_hours = int(hours % 24)
            if remaining_hours == 0:
                return f"{days} days"
            else:
                return f"{days}d {remaining_hours}h"
    
    @staticmethod
    def create_download_link(df: pd.DataFrame, filename: str, 
                           link_text: str = "Download CSV") -> str:
        """
        Create download link for DataFrame
        
        Args:
            df: DataFrame to download
            filename: Name of the file
            link_text: Text for the download link
            
        Returns:
            str: HTML download link
        """
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
        return href
    
    @log_performance
    def extract_email_insights(self, email_text: str) -> Dict[str, Any]:
        """
        Extract insights from email text using NLP techniques
        
        Args:
            email_text: Email content to analyze
            
        Returns:
            Dict[str, Any]: Extracted insights
        """
        self.log_method_entry("extract_email_insights")
        
        if pd.isna(email_text) or email_text == 'nan':
            return {
                'word_count': 0,
                'sentence_count': 0,
                'question_count': 0,
                'urgency_indicators': [],
                'action_items': [],
                'business_terms': [],
                'sentiment_indicators': []
            }
        
        text = str(email_text).lower()
        
        insights = {
            'word_count': len(text.split()),
            'sentence_count': len(re.findall(r'[.!?]+', text)),
            'question_count': text.count('?'),
            'urgency_indicators': self._find_urgency_indicators(text),
            'action_items': self._extract_action_items(text),
            'business_terms': self._find_business_terms(text),
            'sentiment_indicators': self._analyze_sentiment_indicators(text)
        }
        
        self.log_method_exit("extract_email_insights", insights)
        return insights
    
    def _find_urgency_indicators(self, text: str) -> List[str]:
        """Find urgency indicators in text"""
        urgency_patterns = [
            'urgent', 'asap', 'immediate', 'priority', 'deadline',
            'rush', 'critical', 'emergency', 'quickly', 'soon'
        ]
        
        found_indicators = []
        for pattern in urgency_patterns:
            if pattern in text:
                found_indicators.append(pattern)
        
        return found_indicators
    
    def _extract_action_items(self, text: str) -> List[str]:
        """Extract potential action items from text"""
        action_patterns = [
            r'(?:please|can you|could you|would you|let\'s|we need to|action item|todo|to do)\s+([^.!?]*)',
            r'(?:follow up|follow-up|next step|action required|need to)\s+([^.!?]*)',
            r'(?:schedule|arrange|organize|prepare|complete|finish)\s+([^.!?]*)'
        ]
        
        action_items = []
        for pattern in action_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            action_items.extend([match.strip() for match in matches if match.strip()])
        
        return action_items[:5]  # Limit to top 5
    
    def _find_business_terms(self, text: str) -> List[str]:
        """Find business-related terms in text"""
        business_terms = []
        
        for category, keywords in self.config.ENGAGEMENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    business_terms.append(keyword)
        
        return list(set(business_terms))  # Remove duplicates
    
    def _analyze_sentiment_indicators(self, text: str) -> List[str]:
        """Analyze sentiment indicators in text"""
        positive_indicators = [
            'excellent', 'great', 'good', 'pleased', 'happy', 'satisfied',
            'successful', 'positive', 'agree', 'perfect', 'wonderful'
        ]
        
        negative_indicators = [
            'concerned', 'issue', 'problem', 'delay', 'disappointed',
            'frustrated', 'difficult', 'challenge', 'risk', 'urgent'
        ]
        
        sentiment = []
        
        for indicator in positive_indicators:
            if indicator in text:
                sentiment.append(f"positive: {indicator}")
        
        for indicator in negative_indicators:
            if indicator in text:
                sentiment.append(f"negative: {indicator}")
        
        return sentiment[:3]  # Limit to top 3
    
    @staticmethod
    def calculate_engagement_score(metrics: Dict[str, float], 
                                 weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate overall engagement score from multiple metrics
        
        Args:
            metrics: Dictionary of metric names and values
            weights: Optional weights for each metric
            
        Returns:
            float: Calculated engagement score
        """
        if not metrics:
            return 0.0
        
        if weights is None:
            weights = {metric: 1.0 for metric in metrics.keys()}
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric, value in metrics.items():
            if metric in weights and not pd.isna(value):
                total_score += value * weights[metric]
                total_weight += weights[metric]
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    @staticmethod
    def detect_data_quality_issues(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect data quality issues in DataFrame
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dict[str, Any]: Data quality report
        """
        if df.empty:
            return {'status': 'empty_dataframe', 'issues': []}
        
        issues = []
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        high_missing = missing_counts[missing_counts > len(df) * 0.5]
        if len(high_missing) > 0:
            issues.append(f"High missing values in columns: {list(high_missing.index)}")
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            issues.append(f"Found {duplicate_count} duplicate rows")
        
        # Check for email format issues
        email_columns = ['Mail Sender', 'Mail Receiver']
        for col in email_columns:
            if col in df.columns:
                email_series = df[col].dropna().astype(str)
                invalid_emails = email_series[~email_series.str.contains('@', na=False)]
                if len(invalid_emails) > 0:
                    issues.append(f"Invalid email formats in {col}: {len(invalid_emails)} records")
        
        # Check for datetime parsing issues
        datetime_columns = ['Meeting Time (BST/GMT)', 'Email Time (BST/GMT)']
        for col in datetime_columns:
            if col in df.columns:
                try:
                    parsed_dates = pd.to_datetime(df[col], errors='coerce')
                    failed_parsing = parsed_dates.isnull().sum()
                    if failed_parsing > 0:
                        issues.append(f"Date parsing failures in {col}: {failed_parsing} records")
                except Exception:
                    issues.append(f"Cannot parse dates in {col}")
        
        return {
            'status': 'issues_found' if issues else 'clean',
            'issues': issues,
            'total_records': len(df),
            'missing_data_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        }
    
    @staticmethod
    def create_summary_statistics(df: pd.DataFrame, 
                                numeric_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create summary statistics for numeric columns
        
        Args:
            df: DataFrame to analyze
            numeric_columns: Specific columns to analyze
            
        Returns:
            pd.DataFrame: Summary statistics
        """
        if df.empty:
            return pd.DataFrame()
        
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            return pd.DataFrame()
        
        summary = df[numeric_columns].describe()
        
        # Add additional statistics
        summary.loc['missing_count'] = df[numeric_columns].isnull().sum()
        summary.loc['missing_percentage'] = (df[numeric_columns].isnull().sum() / len(df)) * 100
        
        return summary.round(2)
    
    @staticmethod
    def generate_time_buckets(start_date: datetime, end_date: datetime, 
                            bucket_type: str = "monthly") -> List[Tuple[datetime, datetime]]:
        """
        Generate time buckets for analysis
        
        Args:
            start_date: Start date
            end_date: End date
            bucket_type: Type of buckets ('daily', 'weekly', 'monthly')
            
        Returns:
            List[Tuple[datetime, datetime]]: List of time bucket ranges
        """
        buckets = []
        current_date = start_date
        
        while current_date < end_date:
            if bucket_type == "daily":
                next_date = current_date + timedelta(days=1)
            elif bucket_type == "weekly":
                next_date = current_date + timedelta(weeks=1)
            elif bucket_type == "monthly":
                if current_date.month == 12:
                    next_date = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    next_date = current_date.replace(month=current_date.month + 1)
            else:
                raise ValueError(f"Unsupported bucket type: {bucket_type}")
            
            buckets.append((current_date, min(next_date, end_date)))
            current_date = next_date
        
        return buckets
    
    @staticmethod
    def render_info_box(title: str, content: str, box_type: str = "info") -> None:
        """
        Render styled information box
        
        Args:
            title: Box title
            content: Box content
            box_type: Type of box ('info', 'warning', 'error', 'success')
        """
        colors = {
            'info': '#3498db',
            'warning': '#f39c12',
            'error': '#e74c3c',
            'success': '#27ae60'
        }
        
        color = colors.get(box_type, colors['info'])
        
        st.markdown(f"""
        <div style="background-color: {color}15; border-left: 5px solid {color}; 
                    padding: 1rem; margin: 1rem 0; border-radius: 0 5px 5px 0;">
            <h4 style="color: {color}; margin: 0 0 0.5rem 0;">{title}</h4>
            <p style="margin: 0; color: #333;">{content}</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_metric_card(title: str, value: str, change: Optional[str] = None,
                          color: str = "#3498db") -> None:
        """
        Create styled metric card
        
        Args:
            title: Metric title
            value: Metric value
            change: Optional change indicator
            color: Card color
        """
        change_html = f'<p style="margin: 0; font-size: 0.9rem; color: #666;">{change}</p>' if change else ''
        
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-top: 4px solid {color};">
            <h3 style="margin: 0; color: {color}; font-size: 2rem;">{value}</h3>
            <p style="margin: 0.5rem 0 0 0; color: #666; font-weight: 500;">{title}</p>
            {change_html}
        </div>
        """, unsafe_allow_html=True)
    
    @log_performance
    def process_email_threads(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process and analyze email threads
        
        Args:
            df: DataFrame with email data
            
        Returns:
            Dict[str, Any]: Thread analysis results
        """
        self.log_method_entry("process_email_threads", shape=df.shape)
        
        if 'Email Subject' not in df.columns:
            return {'threads': [], 'thread_count': 0, 'avg_thread_length': 0}
        
        # Group emails by similar subjects (thread detection)
        threads = {}
        
        for _, row in df.iterrows():
            subject = str(row.get('Email Subject', '')).strip()
            
            # Clean subject for thread matching
            clean_subject = re.sub(r'^(re:|fwd:|fw:)\s*', '', subject, flags=re.IGNORECASE)
            clean_subject = re.sub(r'\s+', ' ', clean_subject).strip()
            
            if clean_subject:
                if clean_subject not in threads:
                    threads[clean_subject] = []
                threads[clean_subject].append(row.to_dict())
        
        # Calculate thread statistics
        thread_lengths = [len(emails) for emails in threads.values()]
        
        results = {
            'threads': threads,
            'thread_count': len(threads),
            'avg_thread_length': np.mean(thread_lengths) if thread_lengths else 0,
            'max_thread_length': max(thread_lengths) if thread_lengths else 0,
            'single_email_threads': sum(1 for length in thread_lengths if length == 1)
        }
        
        self.logger.info("Processed %d email threads with avg length %.2f", 
                        results['thread_count'], results['avg_thread_length'])
        
        self.log_method_exit("process_email_threads", results)
        return results
    
    @staticmethod
    def validate_excel_structure(df: pd.DataFrame, required_columns: List[str]) -> Dict[str, Any]:
        """
        Validate Excel file structure against requirements
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            Dict[str, Any]: Validation results
        """
        validation_result = {
            'is_valid': True,
            'missing_columns': [],
            'extra_columns': [],
            'data_types': {},
            'recommendations': []
        }
        
        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_result['missing_columns'] = missing_columns
        
        if missing_columns:
            validation_result['is_valid'] = False
            validation_result['recommendations'].append(
                f"Add missing columns: {', '.join(missing_columns)}"
            )
        
        # Check for extra columns
        extra_columns = [col for col in df.columns if col not in required_columns]
        validation_result['extra_columns'] = extra_columns
        
        # Analyze data types
        for col in df.columns:
            validation_result['data_types'][col] = str(df[col].dtype)
        
        # Check data quality
        if df.empty:
            validation_result['is_valid'] = False
            validation_result['recommendations'].append("Excel file is empty")
        
        if len(df) < 10:
            validation_result['recommendations'].append(
                "Dataset is very small - consider adding more data for better analysis"
            )
        
        return validation_result
    
    @staticmethod
    def export_analysis_report(analysis_results: Dict[str, Any], 
                             filename: str = "analysis_report") -> bytes:
        """
        Export analysis results to Excel file
        
        Args:
            analysis_results: Dictionary with analysis results
            filename: Base filename for export
            
        Returns:
            bytes: Excel file content
        """
        # Create Excel writer
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = []
            for key, value in analysis_results.items():
                if isinstance(value, (int, float, str)):
                    summary_data.append({'Metric': key, 'Value': value})
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Add detailed data sheets if available
            for key, value in analysis_results.items():
                if isinstance(value, pd.DataFrame) and not value.empty:
                    sheet_name = key[:31]  # Excel sheet name limit
                    value.to_excel(writer, sheet_name=sheet_name, index=False)
        
        return output.getvalue()
    
    def get_color_palette(self, palette_name: str = "primary") -> List[str]:
        """
        Get color palette for visualizations
        
        Args:
            palette_name: Name of the color palette
            
        Returns:
            List[str]: List of color codes
        """
        return self.config.COLOR_SCHEMES.get(palette_name, self.config.COLOR_SCHEMES['primary'])