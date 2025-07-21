"""
Base Report Class
Abstract base class for all dashboard reports with common functionality
"""

from abc import ABC, abstractmethod
import pandas as pd
import streamlit as st
from typing import Dict, Any, List, Optional
import plotly.express as px
import plotly.graph_objects as go

from logger import get_logger, LoggerMixin, log_performance
from config import DashboardConfig


class BaseReport(ABC, LoggerMixin):
    """
    Abstract base class for all dashboard reports
    
    Provides common functionality and enforces interface consistency
    across all report implementations
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize base report
        
        Args:
            data: Processed DataFrame for analysis
        """
        self.data = data
        self.config = DashboardConfig()
        self.logger.info("Initialized %s with %d records", self.__class__.__name__, len(data))
    
    @abstractmethod
    def generate_analysis(self) -> Dict[str, Any]:
        """
        Generate the core analysis for this report
        
        Returns:
            Dict[str, Any]: Analysis results with metrics and data
        """
        pass
    
    @abstractmethod
    def render_report(self) -> None:
        """
        Render the complete report in Streamlit
        """
        pass
    
    @abstractmethod
    def get_report_summary(self) -> Dict[str, Any]:
        """
        Get a summary of key metrics for this report
        
        Returns:
            Dict[str, Any]: Summary metrics
        """
        pass
    
    def validate_data(self) -> bool:
        """
        Validate that the data contains required columns for this report
        
        Returns:
            bool: True if data is valid for this report
        """
        required_columns = self.get_required_columns()
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            self.logger.error("Missing required columns for %s: %s", 
                            self.__class__.__name__, missing_columns)
            return False
        
        if self.data.empty:
            self.logger.error("Empty dataset provided to %s", self.__class__.__name__)
            return False
        
        return True
    
    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """
        Get list of required columns for this report
        
        Returns:
            List[str]: Required column names
        """
        pass
    
    def render_header(self, title: str, description: str, icon: str = "📊") -> None:
        """
        Render standardized report header
        
        Args:
            title: Report title
            description: Report description
            icon: Report icon
        """
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 10px; margin-bottom: 2rem; color: white;">
            <h1 style="margin: 0; font-size: 2.5rem;">{icon} {title}</h1>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">{description}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_metrics_grid(self, metrics: Dict[str, Any], columns: int = 4) -> None:
        """
        Render metrics in a grid layout
        
        Args:
            metrics: Dictionary of metric_name: value pairs
            columns: Number of columns in the grid
        """
        cols = st.columns(columns)
        
        for i, (metric_name, metric_value) in enumerate(metrics.items()):
            with cols[i % columns]:
                self._render_metric_card(metric_name, metric_value)
    
    def _render_metric_card(self, title: str, value: Any, delta: Optional[str] = None) -> None:
        """
        Render individual metric card
        
        Args:
            title: Metric title
            value: Metric value
            delta: Optional delta value
        """
        st.metric(
            label=title,
            value=value,
            delta=delta
        )
    
    def create_basic_chart(self, chart_type: str, data: pd.DataFrame, 
                          x: str, y: str, **kwargs) -> go.Figure:
        """
        Create basic Plotly charts with consistent styling
        
        Args:
            chart_type: Type of chart ('bar', 'line', 'scatter', 'pie')
            data: Data for the chart
            x: X-axis column
            y: Y-axis column
            **kwargs: Additional plotly arguments
            
        Returns:
            go.Figure: Plotly figure
        """
        self.log_method_entry("create_basic_chart", chart_type=chart_type, shape=data.shape)
        
        # Set default styling
        default_kwargs = {
            'color_discrete_sequence': self.config.COLOR_SCHEMES['primary'],
            'template': 'plotly_white'
        }
        default_kwargs.update(kwargs)
        
        # Create chart based on type
        if chart_type == 'bar':
            fig = px.bar(data, x=x, y=y, **default_kwargs)
        elif chart_type == 'line':
            fig = px.line(data, x=x, y=y, **default_kwargs)
        elif chart_type == 'scatter':
            fig = px.scatter(data, x=x, y=y, **default_kwargs)
        elif chart_type == 'pie':
            fig = px.pie(data, names=x, values=y, **default_kwargs)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        
        # Apply consistent styling
        fig.update_layout(
            font=dict(size=12),
            showlegend=True,
            height=400,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        self.log_method_exit("create_basic_chart", fig)
        return fig
    
    def filter_data_by_timeframe(self, timeframe: str = "All") -> pd.DataFrame:
        """
        Filter data by specified timeframe
        
        Args:
            timeframe: Time period to filter ('All', 'Last 30 days', 'Last 90 days', etc.)
            
        Returns:
            pd.DataFrame: Filtered data
        """
        if timeframe == "All" or 'Meeting Time (BST/GMT)' not in self.data.columns:
            return self.data
        
        # Convert meeting time to datetime if not already
        meeting_time_col = self.data['Meeting Time (BST/GMT)']
        if not pd.api.types.is_datetime64_any_dtype(meeting_time_col):
            meeting_time_col = pd.to_datetime(meeting_time_col, errors='coerce')
        
        # Calculate cutoff date
        from datetime import datetime, timedelta
        now = datetime.now()
        
        if timeframe == "Last 30 days":
            cutoff = now - timedelta(days=30)
        elif timeframe == "Last 90 days":
            cutoff = now - timedelta(days=90)
        elif timeframe == "Last 6 months":
            cutoff = now - timedelta(days=180)
        elif timeframe == "Last year":
            cutoff = now - timedelta(days=365)
        else:
            return self.data
        
        # Filter data
        filtered_data = self.data[meeting_time_col >= cutoff]
        
        self.logger.info("Filtered data from %d to %d records for timeframe: %s", 
                        len(self.data), len(filtered_data), timeframe)
        
        return filtered_data
    
    def export_results(self, results: Dict[str, Any], filename: str) -> None:
        """
        Export report results to various formats
        
        Args:
            results: Analysis results to export
            filename: Base filename for export
        """
        try:
            # Convert results to DataFrame if possible
            if 'data' in results and isinstance(results['data'], pd.DataFrame):
                export_data = results['data']
            else:
                # Create DataFrame from metrics
                export_data = pd.DataFrame([results])
            
            # Create download button
            csv_data = export_data.to_csv(index=False)
            st.download_button(
                label=f"📥 Download {filename}.csv",
                data=csv_data,
                file_name=f"{filename}.csv",
                mime="text/csv"
            )
            
            self.logger.info("Export prepared for %s", filename)
            
        except Exception as e:
            self.logger.error("Failed to prepare export for %s: %s", filename, str(e))
            st.error(f"Failed to prepare export: {str(e)}")
    
    def render_data_quality_warning(self) -> None:
        """
        Render data quality warnings if issues are detected
        """
        warnings = []
        
        # Check for missing data in key columns
        required_cols = self.get_required_columns()
        for col in required_cols:
            if col in self.data.columns:
                null_rate = self.data[col].isnull().mean()
                if null_rate > 0.1:  # More than 10% missing
                    warnings.append(f"Column '{col}' has {null_rate:.1%} missing values")
        
        # Check data recency
        if 'Meeting Time (BST/GMT)' in self.data.columns:
            latest_date = pd.to_datetime(self.data['Meeting Time (BST/GMT)'], errors='coerce').max()
            if pd.notna(latest_date):
                days_since_latest = (pd.Timestamp.now() - latest_date).days
                if days_since_latest > 30:
                    warnings.append(f"Latest data is {days_since_latest} days old")
        
        # Display warnings if any
        if warnings:
            st.warning("⚠️ **Data Quality Issues Detected:**\n" + "\n".join(f"• {w}" for w in warnings))
    
    @log_performance
    def run_report(self) -> None:
        """
        Main method to run the complete report
        """
        self.log_method_entry("run_report")
        
        try:
            # Validate data first
            if not self.validate_data():
                st.error("Data validation failed for this report")
                return
            
            # Render data quality warnings
            self.render_data_quality_warning()
            
            # Generate analysis
            with st.spinner("Generating analysis..."):
                analysis_results = self.generate_analysis()
            
            # Render the report
            self.render_report()
            
            # Provide export option
            report_name = self.__class__.__name__.replace('Report', '').lower()
            self.export_results(analysis_results, f"{report_name}_report")
            
            self.logger.info("Report execution completed successfully")
            self.log_method_exit("run_report")
            
        except Exception as e:
            error_msg = f"Error running report {self.__class__.__name__}: {str(e)}"
            self.logger.error(error_msg)
            st.error(error_msg)
            raise