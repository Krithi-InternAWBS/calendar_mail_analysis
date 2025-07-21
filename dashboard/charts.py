"""
Chart Generation Utilities
Provides standardized chart creation and styling for the dashboard
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import streamlit as st

from logger import get_logger, LoggerMixin, log_performance
from config import DashboardConfig


class ChartGenerator(LoggerMixin):
    """
    Utility class for generating standardized Plotly charts for the dashboard
    """
    
    def __init__(self):
        """Initialize chart generator with configuration"""
        self.config = DashboardConfig()
        self.logger.info("ChartGenerator initialized")
        
        # Default chart settings
        self.default_layout = {
            'template': 'plotly_white',
            'font': {'size': 12, 'family': 'Arial'},
            'showlegend': True,
            'height': 400,
            'margin': {'l': 50, 'r': 50, 't': 60, 'b': 50}
        }
    
    @log_performance
    def create_engagement_pie_chart(self, engagement_data: Dict[str, int], 
                                   title: str = "Engagement Distribution") -> go.Figure:
        """
        Create pie chart for engagement analysis
        
        Args:
            engagement_data: Dictionary with category names and counts
            title: Chart title
            
        Returns:
            go.Figure: Plotly pie chart
        """
        self.log_method_entry("create_engagement_pie_chart", title=title)
        
        categories = list(engagement_data.keys())
        values = list(engagement_data.values())
        
        fig = px.pie(
            names=categories,
            values=values,
            title=title,
            color_discrete_sequence=self.config.COLOR_SCHEMES['engagement']
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
        
        fig.update_layout(**self.default_layout)
        
        self.log_method_exit("create_engagement_pie_chart", fig)
        return fig
    
    @log_performance
    def create_response_time_histogram(self, response_times: pd.Series, 
                                     title: str = "Response Time Distribution") -> go.Figure:
        """
        Create histogram for response time analysis
        
        Args:
            response_times: Series of response times in hours
            title: Chart title
            
        Returns:
            go.Figure: Plotly histogram
        """
        self.log_method_entry("create_response_time_histogram", title=title)
        
        fig = px.histogram(
            response_times,
            nbins=20,
            title=title,
            labels={'value': 'Response Time (hours)', 'count': 'Number of Responses'},
            color_discrete_sequence=self.config.COLOR_SCHEMES['response_time']
        )
        
        # Add threshold lines
        thresholds = [
            (1, "1 hour", "green"),
            (24, "24 hours", "orange"), 
            (48, "48 hours", "red")
        ]
        
        for threshold, label, color in thresholds:
            fig.add_vline(
                x=threshold,
                line_dash="dash",
                line_color=color,
                annotation_text=label,
                annotation_position="top right"
            )
        
        fig.update_layout(**self.default_layout)
        
        self.log_method_exit("create_response_time_histogram", fig)
        return fig
    
    @log_performance
    def create_monthly_trend_chart(self, monthly_data: pd.DataFrame, 
                                 metrics: List[str],
                                 title: str = "Monthly Activity Trends") -> go.Figure:
        """
        Create line chart for monthly trends analysis
        
        Args:
            monthly_data: DataFrame with monthly data
            metrics: List of metric columns to plot
            title: Chart title
            
        Returns:
            go.Figure: Plotly line chart
        """
        self.log_method_entry("create_monthly_trend_chart", title=title, metrics=metrics)
        
        fig = go.Figure()
        
        # Convert index to string for plotting
        x_values = monthly_data.index.astype(str) if hasattr(monthly_data.index, 'astype') else monthly_data.index
        
        colors = self.config.COLOR_SCHEMES['primary']
        
        for i, metric in enumerate(metrics):
            if metric in monthly_data.columns:
                color = colors[i % len(colors)]
                
                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=monthly_data[metric],
                    mode='lines+markers',
                    name=metric.replace('_', ' ').title(),
                    line=dict(color=color, width=3),
                    marker=dict(size=8)
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Month",
            yaxis_title="Count",
            hovermode='x unified',
            **self.default_layout
        )
        
        self.log_method_exit("create_monthly_trend_chart", fig)
        return fig
    
    @log_performance
    def create_client_engagement_matrix(self, matrix_data: pd.DataFrame,
                                      title: str = "Client Engagement Matrix") -> go.Figure:
        """
        Create heatmap for client engagement analysis
        
        Args:
            matrix_data: DataFrame with client engagement data
            title: Chart title
            
        Returns:
            go.Figure: Plotly heatmap
        """
        self.log_method_entry("create_client_engagement_matrix", title=title)
        
        # Prepare data for heatmap
        if matrix_data.empty:
            # Create empty heatmap
            fig = go.Figure(data=go.Heatmap(
                z=[[0]],
                x=['No Data'],
                y=['No Data'],
                colorscale='Blues'
            ))
        else:
            # Select numeric columns for heatmap
            numeric_cols = matrix_data.select_dtypes(include=[np.number]).columns
            heatmap_data = matrix_data[numeric_cols].head(20)  # Top 20 clients
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='Blues',
                showscale=True,
                hovertemplate='Client: %{y}<br>Metric: %{x}<br>Value: %{z}<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Engagement Metrics",
            yaxis_title="Client Domains",
            **self.default_layout
        )
        
        self.log_method_exit("create_client_engagement_matrix", fig)
        return fig
    
    @log_performance
    def create_keyword_frequency_chart(self, keyword_data: Dict[str, int],
                                     title: str = "Top Keywords",
                                     top_n: int = 15) -> go.Figure:
        """
        Create horizontal bar chart for keyword frequency
        
        Args:
            keyword_data: Dictionary with keywords and frequencies
            title: Chart title
            top_n: Number of top keywords to show
            
        Returns:
            go.Figure: Plotly bar chart
        """
        self.log_method_entry("create_keyword_frequency_chart", title=title, top_n=top_n)
        
        # Sort and get top N keywords
        sorted_keywords = sorted(keyword_data.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        keywords = [item[0] for item in sorted_keywords]
        frequencies = [item[1] for item in sorted_keywords]
        
        fig = px.bar(
            x=frequencies,
            y=keywords,
            orientation='h',
            title=title,
            labels={'x': 'Frequency', 'y': 'Keywords'},
            color_discrete_sequence=self.config.COLOR_SCHEMES['primary']
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            **self.default_layout
        )
        
        self.log_method_exit("create_keyword_frequency_chart", fig)
        return fig
    
    @log_performance
    def create_correlation_scatter_plot(self, data: pd.DataFrame, x_col: str, y_col: str,
                                      title: str = "Correlation Analysis",
                                      color_col: Optional[str] = None) -> go.Figure:
        """
        Create scatter plot for correlation analysis
        
        Args:
            data: DataFrame with data to plot
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            title: Chart title
            color_col: Optional column for color coding
            
        Returns:
            go.Figure: Plotly scatter plot
        """
        self.log_method_entry("create_correlation_scatter_plot", title=title, 
                             x_col=x_col, y_col=y_col)
        
        if data.empty or x_col not in data.columns or y_col not in data.columns:
            # Create empty plot
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for correlation analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        else:
            # Create scatter plot
            fig = px.scatter(
                data,
                x=x_col,
                y=y_col,
                color=color_col if color_col and color_col in data.columns else None,
                title=title,
                color_discrete_sequence=self.config.COLOR_SCHEMES['primary']
            )
            
            # Add trend line
            if len(data) > 1 and data[x_col].notna().sum() > 1 and data[y_col].notna().sum() > 1:
                # Calculate correlation coefficient
                correlation = data[x_col].corr(data[y_col])
                
                # Add trend line
                z = np.polyfit(data[x_col].dropna(), data[y_col].dropna(), 1)
                p = np.poly1d(z)
                x_trend = np.linspace(data[x_col].min(), data[x_col].max(), 100)
                
                fig.add_trace(go.Scatter(
                    x=x_trend,
                    y=p(x_trend),
                    mode='lines',
                    name=f'Trend (r={correlation:.3f})',
                    line=dict(dash='dash', color='red')
                ))
        
        fig.update_layout(
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title(),
            **self.default_layout
        )
        
        self.log_method_exit("create_correlation_scatter_plot", fig)
        return fig
    
    @log_performance
    def create_time_series_chart(self, data: pd.DataFrame, date_col: str, value_col: str,
                               title: str = "Time Series Analysis",
                               rolling_window: Optional[int] = None) -> go.Figure:
        """
        Create time series chart with optional rolling average
        
        Args:
            data: DataFrame with time series data
            date_col: Column name for dates
            value_col: Column name for values
            title: Chart title
            rolling_window: Optional rolling window size for moving average
            
        Returns:
            go.Figure: Plotly time series chart
        """
        self.log_method_entry("create_time_series_chart", title=title, 
                             date_col=date_col, value_col=value_col)
        
        if data.empty or date_col not in data.columns or value_col not in data.columns:
            fig = go.Figure()
            fig.add_annotation(
                text="No time series data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        else:
            # Sort by date
            data_sorted = data.sort_values(date_col)
            
            fig = go.Figure()
            
            # Add main time series
            fig.add_trace(go.Scatter(
                x=data_sorted[date_col],
                y=data_sorted[value_col],
                mode='lines+markers',
                name='Actual',
                line=dict(color=self.config.COLOR_SCHEMES['trend'][0])
            ))
            
            # Add rolling average if requested
            if rolling_window and len(data_sorted) >= rolling_window:
                rolling_avg = data_sorted[value_col].rolling(window=rolling_window).mean()
                fig.add_trace(go.Scatter(
                    x=data_sorted[date_col],
                    y=rolling_avg,
                    mode='lines',
                    name=f'{rolling_window}-period Moving Average',
                    line=dict(color=self.config.COLOR_SCHEMES['trend'][1], dash='dash')
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=value_col.replace('_', ' ').title(),
            **self.default_layout
        )
        
        self.log_method_exit("create_time_series_chart", fig)
        return fig
    
    @log_performance
    def create_multi_metric_dashboard(self, metrics_data: Dict[str, Any],
                                    title: str = "Multi-Metric Dashboard") -> go.Figure:
        """
        Create dashboard-style chart with multiple metrics
        
        Args:
            metrics_data: Dictionary with metric names and values
            title: Chart title
            
        Returns:
            go.Figure: Plotly subplot figure
        """
        self.log_method_entry("create_multi_metric_dashboard", title=title)
        
        # Create subplots
        num_metrics = len(metrics_data)
        cols = min(2, num_metrics)
        rows = (num_metrics + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=list(metrics_data.keys()),
            specs=[[{"type": "indicator"}] * cols for _ in range(rows)]
        )
        
        for i, (metric_name, metric_value) in enumerate(metrics_data.items()):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            # Determine value format and color
            if isinstance(metric_value, (int, float)):
                if isinstance(metric_value, float):
                    value_text = f"{metric_value:.2f}"
                else:
                    value_text = f"{metric_value:,}"
            else:
                value_text = str(metric_value)
            
            # Add indicator
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=metric_value if isinstance(metric_value, (int, float)) else 0,
                    title={"text": metric_name},
                    number={'font': {'size': 40}},
                    domain={'row': row-1, 'column': col-1}
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title=title,
            height=200 * rows,
            **{k: v for k, v in self.default_layout.items() if k != 'height'}
        )
        
        self.log_method_exit("create_multi_metric_dashboard", fig)
        return fig
    
    @log_performance
    def create_comparison_bar_chart(self, comparison_data: pd.DataFrame,
                                   category_col: str, value_col: str,
                                   title: str = "Comparison Analysis",
                                   orientation: str = 'v') -> go.Figure:
        """
        Create bar chart for comparing categories
        
        Args:
            comparison_data: DataFrame with comparison data
            category_col: Column name for categories
            value_col: Column name for values
            title: Chart title
            orientation: 'v' for vertical, 'h' for horizontal
            
        Returns:
            go.Figure: Plotly bar chart
        """
        self.log_method_entry("create_comparison_bar_chart", title=title, 
                             category_col=category_col, value_col=value_col)
        
        if comparison_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No comparison data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        else:
            # Sort data by value
            sorted_data = comparison_data.sort_values(value_col, ascending=(orientation == 'h'))
            
            if orientation == 'h':
                fig = px.bar(
                    sorted_data,
                    x=value_col,
                    y=category_col,
                    orientation='h',
                    title=title,
                    color_discrete_sequence=self.config.COLOR_SCHEMES['primary']
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            else:
                fig = px.bar(
                    sorted_data,
                    x=category_col,
                    y=value_col,
                    title=title,
                    color_discrete_sequence=self.config.COLOR_SCHEMES['primary']
                )
        
        fig.update_layout(**self.default_layout)
        
        self.log_method_exit("create_comparison_bar_chart", fig)
        return fig
    
    def apply_custom_styling(self, fig: go.Figure, style_config: Dict[str, Any]) -> go.Figure:
        """
        Apply custom styling to a Plotly figure
        
        Args:
            fig: Plotly figure to style
            style_config: Dictionary with styling options
            
        Returns:
            go.Figure: Styled figure
        """
        self.log_method_entry("apply_custom_styling")
        
        # Update layout with custom styles
        fig.update_layout(**style_config)
        
        # Apply consistent color scheme if not specified
        if 'colorway' not in style_config:
            fig.update_layout(colorway=self.config.COLOR_SCHEMES['primary'])
        
        self.log_method_exit("apply_custom_styling", fig)
        return fig
    
    def create_empty_chart_placeholder(self, message: str = "No data available") -> go.Figure:
        """
        Create placeholder chart when no data is available
        
        Args:
            message: Message to display
            
        Returns:
            go.Figure: Empty placeholder chart
        """
        fig = go.Figure()
        
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            **self.default_layout
        )
        
        return fig
    
    def save_chart_config(self, chart_type: str, config: Dict[str, Any]) -> None:
        """
        Save chart configuration for reuse
        
        Args:
            chart_type: Type of chart
            config: Configuration dictionary
        """
        if not hasattr(self, 'saved_configs'):
            self.saved_configs = {}
        
        self.saved_configs[chart_type] = config
        self.logger.info("Saved configuration for chart type: %s", chart_type)
    
    def load_chart_config(self, chart_type: str) -> Dict[str, Any]:
        """
        Load saved chart configuration
        
        Args:
            chart_type: Type of chart
            
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        if hasattr(self, 'saved_configs') and chart_type in self.saved_configs:
            return self.saved_configs[chart_type]
        
        return self.default_layout.copy()