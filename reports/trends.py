"""
Monthly Activity Trend Report
Shows volume of meetings and correlated emails month by month
"""

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import calendar

from .base import BaseReport
from logger import log_performance


class TrendsReport(BaseReport):
    """
    Report 5: Monthly Activity Trend Report
    
    Purpose: Show volume of meetings and correlated emails month by month, 
    to detect periods of low engagement or "invisible work."
    Output: Graph of meetings, emails sent, and correlation ratio
    Value: Tracks consistency, effort over time, potential ghost billing
    """
    
    def get_required_columns(self) -> List[str]:
        """Get required columns for trends analysis"""
        return [
            'Meeting Time (BST/GMT)',
            'Email Time (BST/GMT)',
            'Meeting Subject',
            'Email Subject',
            'Direction'
        ]
    
    @log_performance
    def generate_analysis(self) -> Dict[str, Any]:
        """
        Generate monthly activity trend analysis
        
        Returns:
            Dict[str, Any]: Analysis results
        """
        self.log_method_entry("generate_analysis")
        
        # Prepare time-based data
        time_series_data = self._prepare_time_series_data()
        
        # Calculate monthly trends
        monthly_trends = self._calculate_monthly_trends(time_series_data)
        
        # Analyze activity patterns
        pattern_analysis = self._analyze_activity_patterns(monthly_trends)
        
        # Detect anomalies and gaps
        anomaly_detection = self._detect_activity_anomalies(monthly_trends)
        
        # Calculate correlation metrics
        correlation_analysis = self._analyze_email_meeting_correlation(monthly_trends)
        
        # Quarter and seasonal analysis
        seasonal_analysis = self._analyze_seasonal_patterns(time_series_data)
        
        results = {
            'time_series_data': time_series_data,
            'monthly_trends': monthly_trends,
            'pattern_analysis': pattern_analysis,
            'anomaly_detection': anomaly_detection,
            'correlation_analysis': correlation_analysis,
            'seasonal_analysis': seasonal_analysis,
            'data_range': self._calculate_data_range(time_series_data)
        }
        
        self.logger.info("Trends analysis completed: %d months of data analyzed",
                        len(monthly_trends) if not monthly_trends.empty else 0)
        
        self.log_method_exit("generate_analysis", results)
        return results
    
    def _prepare_time_series_data(self) -> pd.DataFrame:
        """
        Prepare and clean time series data
        
        Returns:
            pd.DataFrame: Cleaned time series data
        """
        self.log_method_entry("_prepare_time_series_data")
        
        df_time = self.data.copy()
        
        # Convert datetime columns
        for col in ['Meeting Time (BST/GMT)', 'Email Time (BST/GMT)']:
            if col in df_time.columns:
                df_time[col] = pd.to_datetime(df_time[col], errors='coerce')
        
        # Create unified activity timeline
        activities = []
        
        # Add meeting activities
        meeting_data = df_time[['Meeting Time (BST/GMT)', 'Meeting Subject']].dropna()
        for _, row in meeting_data.iterrows():
            activities.append({
                'Date': row['Meeting Time (BST/GMT)'],
                'Activity_Type': 'Meeting',
                'Subject': row['Meeting Subject'],
                'Direction': 'Meeting'
            })
        
        # Add email activities
        email_data = df_time[['Email Time (BST/GMT)', 'Email Subject', 'Direction']].dropna()
        for _, row in email_data.iterrows():
            activities.append({
                'Date': row['Email Time (BST/GMT)'],
                'Activity_Type': 'Email',
                'Subject': row['Email Subject'],
                'Direction': row['Direction']
            })
        
        # Create DataFrame
        activity_df = pd.DataFrame(activities)
        
        if len(activity_df) > 0:
            # Add time components
            activity_df['Month'] = activity_df['Date'].dt.to_period('M')
            activity_df['Quarter'] = activity_df['Date'].dt.to_period('Q')
            activity_df['Week'] = activity_df['Date'].dt.to_period('W')
            activity_df['DayOfWeek'] = activity_df['Date'].dt.dayofweek
            activity_df['Hour'] = activity_df['Date'].dt.hour
        
        self.logger.info("Prepared time series data with %d activities", len(activity_df))
        self.log_method_exit("_prepare_time_series_data", activity_df)
        
        return activity_df
    
    def _calculate_monthly_trends(self, activity_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate monthly activity trends
        
        Args:
            activity_df: Time series activity data
            
        Returns:
            pd.DataFrame: Monthly trend metrics
        """
        self.log_method_entry("_calculate_monthly_trends", shape=activity_df.shape)
        
        if activity_df.empty:
            return pd.DataFrame()
        
        # Group by month and activity type
        monthly_summary = activity_df.groupby(['Month', 'Activity_Type']).agg({
            'Subject': 'nunique',  # Unique subjects
            'Date': 'count'        # Total activities
        }).unstack(fill_value=0)
        
        # Flatten column names
        monthly_summary.columns = [
            f"{col[1]}_{col[0]}" for col in monthly_summary.columns
        ]
        
        # Rename columns for clarity
        column_mapping = {
            'Email_Subject': 'Unique_Emails',
            'Email_Date': 'Total_Email_Activities',
            'Meeting_Subject': 'Unique_Meetings', 
            'Meeting_Date': 'Total_Meeting_Activities'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in monthly_summary.columns:
                monthly_summary[new_col] = monthly_summary[old_col]
        
        # Fill missing columns with zeros
        for col in ['Unique_Emails', 'Total_Email_Activities', 'Unique_Meetings', 'Total_Meeting_Activities']:
            if col not in monthly_summary.columns:
                monthly_summary[col] = 0
        
        # Calculate correlation ratio (emails per meeting)
        monthly_summary['Correlation_Ratio'] = (
            monthly_summary['Total_Email_Activities'] / 
            monthly_summary['Total_Meeting_Activities'].replace(0, 1)
        ).replace([float('inf'), -float('inf')], 0).fillna(0)
        
        # Calculate activity density
        monthly_summary['Total_Activity'] = (
            monthly_summary['Total_Email_Activities'] + 
            monthly_summary['Total_Meeting_Activities']
        )
        
        # Calculate unique activity ratio
        monthly_summary['Unique_Activity_Ratio'] = (
            monthly_summary['Unique_Emails'] + monthly_summary['Unique_Meetings']
        ) / monthly_summary['Total_Activity'].replace(0, 1)
        
        # Add rolling averages (3-month)
        for col in ['Total_Activity', 'Correlation_Ratio', 'Unique_Meetings', 'Unique_Emails']:
            monthly_summary[f'{col}_3M_Avg'] = monthly_summary[col].rolling(
                window=3, min_periods=1
            ).mean()
        
        self.logger.info("Calculated monthly trends for %d months", len(monthly_summary))
        self.log_method_exit("_calculate_monthly_trends", monthly_summary)
        
        return monthly_summary
    
    def _analyze_activity_patterns(self, monthly_trends: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze activity patterns and trends
        
        Args:
            monthly_trends: Monthly trend data
            
        Returns:
            Dict[str, Any]: Pattern analysis
        """
        if monthly_trends.empty:
            return {}
        
        patterns = {}
        
        # Overall trend direction
        if len(monthly_trends) >= 2:
            # Calculate trend slopes
            months_numeric = np.arange(len(monthly_trends))
            
            activity_slope = np.polyfit(months_numeric, monthly_trends['Total_Activity'], 1)[0]
            correlation_slope = np.polyfit(months_numeric, monthly_trends['Correlation_Ratio'], 1)[0]
            
            patterns['activity_trend'] = 'Increasing' if activity_slope > 0 else 'Decreasing' if activity_slope < 0 else 'Stable'
            patterns['correlation_trend'] = 'Improving' if correlation_slope > 0 else 'Declining' if correlation_slope < 0 else 'Stable'
            patterns['activity_slope'] = activity_slope
            patterns['correlation_slope'] = correlation_slope
        
        # Identify peak and low periods
        if 'Total_Activity' in monthly_trends.columns:
            patterns['peak_activity_month'] = monthly_trends['Total_Activity'].idxmax()
            patterns['low_activity_month'] = monthly_trends['Total_Activity'].idxmin()
            patterns['peak_activity_value'] = monthly_trends['Total_Activity'].max()
            patterns['low_activity_value'] = monthly_trends['Total_Activity'].min()
        
        # Variability analysis
        patterns['activity_variability'] = monthly_trends['Total_Activity'].std()
        patterns['correlation_variability'] = monthly_trends['Correlation_Ratio'].std()
        
        # Consistency metrics
        patterns['avg_monthly_activity'] = monthly_trends['Total_Activity'].mean()
        patterns['avg_correlation_ratio'] = monthly_trends['Correlation_Ratio'].mean()
        
        return patterns
    
    def _detect_activity_anomalies(self, monthly_trends: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect anomalies and unusual patterns in activity
        
        Args:
            monthly_trends: Monthly trend data
            
        Returns:
            Dict[str, Any]: Anomaly detection results
        """
        if monthly_trends.empty:
            return {}
        
        anomalies = {}
        
        # Statistical anomaly detection using IQR method
        def detect_outliers(series):
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return series[(series < lower_bound) | (series > upper_bound)]
        
        # Detect activity anomalies
        if 'Total_Activity' in monthly_trends.columns:
            activity_outliers = detect_outliers(monthly_trends['Total_Activity'])
            anomalies['activity_outliers'] = activity_outliers.to_dict()
            anomalies['low_activity_months'] = activity_outliers[activity_outliers < monthly_trends['Total_Activity'].median()].to_dict()
            anomalies['high_activity_months'] = activity_outliers[activity_outliers > monthly_trends['Total_Activity'].median()].to_dict()
        
        # Detect correlation anomalies
        if 'Correlation_Ratio' in monthly_trends.columns:
            correlation_outliers = detect_outliers(monthly_trends['Correlation_Ratio'])
            anomalies['correlation_outliers'] = correlation_outliers.to_dict()
        
        # Identify periods of "invisible work" (high meetings, low emails)
        if 'Unique_Meetings' in monthly_trends.columns and 'Unique_Emails' in monthly_trends.columns:
            invisible_work = monthly_trends[
                (monthly_trends['Unique_Meetings'] > monthly_trends['Unique_Meetings'].median()) &
                (monthly_trends['Unique_Emails'] < monthly_trends['Unique_Emails'].median())
            ]
            anomalies['invisible_work_periods'] = invisible_work.index.tolist()
        
        # Detect sudden drops (>50% decrease from previous month)
        for col in ['Total_Activity', 'Unique_Meetings', 'Unique_Emails']:
            if col in monthly_trends.columns:
                pct_change = monthly_trends[col].pct_change()
                sudden_drops = pct_change[pct_change < -0.5]
                anomalies[f'{col}_sudden_drops'] = sudden_drops.to_dict()
        
        return anomalies
    
    def _analyze_email_meeting_correlation(self, monthly_trends: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze correlation between email and meeting activities
        
        Args:
            monthly_trends: Monthly trend data
            
        Returns:
            Dict[str, Any]: Correlation analysis
        """
        if monthly_trends.empty:
            return {}
        
        correlation_metrics = {}
        
        # Pearson correlation between meetings and emails
        if 'Unique_Meetings' in monthly_trends.columns and 'Unique_Emails' in monthly_trends.columns:
            correlation_metrics['meeting_email_correlation'] = monthly_trends['Unique_Meetings'].corr(
                monthly_trends['Unique_Emails']
            )
        
        # Correlation ratio statistics
        if 'Correlation_Ratio' in monthly_trends.columns:
            correlation_metrics['avg_correlation_ratio'] = monthly_trends['Correlation_Ratio'].mean()
            correlation_metrics['correlation_ratio_std'] = monthly_trends['Correlation_Ratio'].std()
            correlation_metrics['correlation_ratio_trend'] = self._calculate_trend_direction(
                monthly_trends['Correlation_Ratio']
            )
        
        # Identify well-correlated vs poorly correlated months
        if 'Correlation_Ratio' in monthly_trends.columns:
            median_ratio = monthly_trends['Correlation_Ratio'].median()
            correlation_metrics['well_correlated_months'] = monthly_trends[
                monthly_trends['Correlation_Ratio'] > median_ratio
            ].index.tolist()
            correlation_metrics['poorly_correlated_months'] = monthly_trends[
                monthly_trends['Correlation_Ratio'] < median_ratio * 0.5
            ].index.tolist()
        
        return correlation_metrics
    
    def _analyze_seasonal_patterns(self, activity_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze seasonal and quarterly patterns
        
        Args:
            activity_df: Time series activity data
            
        Returns:
            Dict[str, Any]: Seasonal analysis
        """
        if activity_df.empty:
            return {}
        
        seasonal = {}
        
        # Quarterly analysis
        if 'Quarter' in activity_df.columns:
            quarterly_activity = activity_df.groupby(['Quarter', 'Activity_Type']).size().unstack(fill_value=0)
            seasonal['quarterly_trends'] = quarterly_activity
        
        # Day of week patterns
        if 'DayOfWeek' in activity_df.columns:
            dow_activity = activity_df.groupby(['DayOfWeek', 'Activity_Type']).size().unstack(fill_value=0)
            # Map numbers to day names
            day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                        4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
            dow_activity.index = dow_activity.index.map(day_names)
            seasonal['day_of_week_patterns'] = dow_activity
        
        # Hourly patterns
        if 'Hour' in activity_df.columns:
            hourly_activity = activity_df.groupby(['Hour', 'Activity_Type']).size().unstack(fill_value=0)
            seasonal['hourly_patterns'] = hourly_activity
        
        return seasonal
    
    def _calculate_data_range(self, activity_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate data range and coverage statistics
        
        Args:
            activity_df: Time series activity data
            
        Returns:
            Dict[str, Any]: Data range information
        """
        if activity_df.empty or 'Date' not in activity_df.columns:
            return {}
        
        dates = activity_df['Date'].dropna()
        
        if len(dates) == 0:
            return {}
        
        return {
            'start_date': dates.min(),
            'end_date': dates.max(),
            'total_days': (dates.max() - dates.min()).days,
            'total_months': len(activity_df['Month'].unique()) if 'Month' in activity_df.columns else 0,
            'data_completeness': len(dates) / ((dates.max() - dates.min()).days + 1) if dates.max() != dates.min() else 1
        }
    
    def _calculate_trend_direction(self, series: pd.Series) -> str:
        """
        Calculate trend direction for a time series
        
        Args:
            series: Time series data
            
        Returns:
            str: Trend direction
        """
        if len(series) < 2:
            return "Insufficient data"
        
        # Simple linear trend
        x = np.arange(len(series))
        slope = np.polyfit(x, series.values, 1)[0]
        
        threshold = series.std() * 0.1  # 10% of standard deviation
        
        if slope > threshold:
            return "Increasing"
        elif slope < -threshold:
            return "Decreasing"
        else:
            return "Stable"
    
    def render_report(self) -> None:
        """Render the complete trends report"""
        self.log_method_entry("render_report")
        
        # Generate analysis
        analysis = self.generate_analysis()
        
        # Render header
        self.render_header(
            "Monthly Activity Trend Report",
            "Tracks meeting and email activity patterns over time to identify engagement trends",
            "📈"
        )
        
        # Render overview metrics
        self._render_overview_metrics(analysis)
        
        # Render trend visualizations
        self._render_trend_charts(analysis)
        
        # Render pattern analysis
        self._render_pattern_analysis(analysis)
        
        # Render insights and anomalies
        self._render_insights_and_anomalies(analysis)
        
        self.log_method_exit("render_report")
    
    def _render_overview_metrics(self, analysis: Dict[str, Any]) -> None:
        """Render overview metrics"""
        st.subheader("📊 Activity Overview")
        
        data_range = analysis.get('data_range', {})
        pattern_analysis = analysis.get('pattern_analysis', {})
        monthly_trends = analysis.get('monthly_trends', pd.DataFrame())
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_months = data_range.get('total_months', 0)
            st.metric(
                "Analysis Period",
                f"{total_months} months",
                help="Total months covered in the analysis"
            )
        
        with col2:
            avg_activity = pattern_analysis.get('avg_monthly_activity', 0)
            st.metric(
                "Avg Monthly Activity",
                f"{avg_activity:.0f}",
                help="Average total activities per month"
            )
        
        with col3:
            avg_correlation = pattern_analysis.get('avg_correlation_ratio', 0)
            st.metric(
                "Avg Email/Meeting Ratio",
                f"{avg_correlation:.2f}",
                help="Average ratio of emails to meetings"
            )
        
        with col4:
            activity_trend = pattern_analysis.get('activity_trend', 'Unknown')
            trend_color = "normal"
            if activity_trend == "Increasing":
                trend_color = "normal"
            elif activity_trend == "Decreasing":
                trend_color = "inverse"
            
            st.metric(
                "Activity Trend",
                activity_trend,
                help="Overall trend direction for activity levels"
            )
        
        # Data range information
        if data_range:
            start_date = data_range.get('start_date')
            end_date = data_range.get('end_date')
            if start_date and end_date:
                st.info(f"📅 **Data Range**: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    def _render_trend_charts(self, analysis: Dict[str, Any]) -> None:
        """Render trend visualization charts"""
        st.subheader("📈 Activity Trends")
        
        monthly_trends = analysis.get('monthly_trends', pd.DataFrame())
        
        if monthly_trends.empty:
            st.warning("⚠️ No trend data available for visualization")
            return
        
        # Main trend chart
        self._render_main_trend_chart(monthly_trends)
        
        # Secondary charts
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_correlation_trend_chart(monthly_trends)
        
        with col2:
            self._render_activity_composition_chart(monthly_trends)
    
    def _render_main_trend_chart(self, monthly_trends: pd.DataFrame) -> None:
        """Render main activity trend chart"""
        # Prepare data for plotting
        trend_data = monthly_trends.reset_index()
        trend_data['Month_Str'] = trend_data['Month'].astype(str)
        
        # Create multi-line chart
        fig = go.Figure()
        
        # Add meeting trend
        if 'Unique_Meetings' in monthly_trends.columns:
            fig.add_trace(go.Scatter(
                x=trend_data['Month_Str'],
                y=trend_data['Unique_Meetings'],
                mode='lines+markers',
                name='Unique Meetings',
                line=dict(color=self.config.COLOR_SCHEMES['primary'][0])
            ))
        
        # Add email trend
        if 'Unique_Emails' in monthly_trends.columns:
            fig.add_trace(go.Scatter(
                x=trend_data['Month_Str'],
                y=trend_data['Unique_Emails'],
                mode='lines+markers',
                name='Unique Emails',
                line=dict(color=self.config.COLOR_SCHEMES['primary'][1])
            ))
        
        # Add total activity
        if 'Total_Activity' in monthly_trends.columns:
            fig.add_trace(go.Scatter(
                x=trend_data['Month_Str'],
                y=trend_data['Total_Activity'],
                mode='lines+markers',
                name='Total Activity',
                line=dict(color=self.config.COLOR_SCHEMES['primary'][2])
            ))
        
        fig.update_layout(
            title="Monthly Activity Trends",
            xaxis_title="Month",
            yaxis_title="Activity Count",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_correlation_trend_chart(self, monthly_trends: pd.DataFrame) -> None:
        """Render correlation ratio trend chart"""
        if 'Correlation_Ratio' not in monthly_trends.columns:
            return
        
        trend_data = monthly_trends.reset_index()
        trend_data['Month_Str'] = trend_data['Month'].astype(str)
        
        fig = px.line(
            trend_data,
            x='Month_Str',
            y='Correlation_Ratio',
            title="Email/Meeting Correlation Ratio",
            color_discrete_sequence=self.config.COLOR_SCHEMES['trend']
        )
        
        # Add average line
        avg_ratio = monthly_trends['Correlation_Ratio'].mean()
        fig.add_hline(y=avg_ratio, line_dash="dash", line_color="red", 
                     annotation_text=f"Average: {avg_ratio:.2f}")
        
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Emails per Meeting",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_activity_composition_chart(self, monthly_trends: pd.DataFrame) -> None:
        """Render activity composition chart"""
        required_cols = ['Unique_Meetings', 'Unique_Emails']
        if not all(col in monthly_trends.columns for col in required_cols):
            return
        
        # Create stacked bar chart
        trend_data = monthly_trends.reset_index()
        trend_data['Month_Str'] = trend_data['Month'].astype(str)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=trend_data['Month_Str'],
            y=trend_data['Unique_Meetings'],
            name='Meetings',
            marker_color=self.config.COLOR_SCHEMES['primary'][0]
        ))
        
        fig.add_trace(go.Bar(
            x=trend_data['Month_Str'],
            y=trend_data['Unique_Emails'],
            name='Emails',
            marker_color=self.config.COLOR_SCHEMES['primary'][1]
        ))
        
        fig.update_layout(
            title="Monthly Activity Composition",
            xaxis_title="Month",
            yaxis_title="Count",
            barmode='stack',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_pattern_analysis(self, analysis: Dict[str, Any]) -> None:
        """Render pattern analysis section"""
        st.subheader("🔍 Pattern Analysis")
        
        pattern_analysis = analysis.get('pattern_analysis', {})
        seasonal_analysis = analysis.get('seasonal_analysis', {})
        
        # Pattern insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Activity Patterns**")
            
            if pattern_analysis:
                trend_direction = pattern_analysis.get('activity_trend', 'Unknown')
                correlation_trend = pattern_analysis.get('correlation_trend', 'Unknown')
                
                st.write(f"• **Activity Trend**: {trend_direction}")
                st.write(f"• **Correlation Trend**: {correlation_trend}")
                
                if 'peak_activity_month' in pattern_analysis:
                    peak_month = pattern_analysis['peak_activity_month']
                    st.write(f"• **Peak Activity**: {peak_month}")
                
                if 'low_activity_month' in pattern_analysis:
                    low_month = pattern_analysis['low_activity_month']
                    st.write(f"• **Lowest Activity**: {low_month}")
        
        with col2:
            # Day of week patterns
            if 'day_of_week_patterns' in seasonal_analysis:
                dow_patterns = seasonal_analysis['day_of_week_patterns']
                if not dow_patterns.empty:
                    fig = px.bar(
                        dow_patterns.reset_index(),
                        x='DayOfWeek',
                        y=dow_patterns.columns.tolist(),
                        title="Activity by Day of Week",
                        color_discrete_sequence=self.config.COLOR_SCHEMES['primary']
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    def _render_insights_and_anomalies(self, analysis: Dict[str, Any]) -> None:
        """Render insights and anomaly detection"""
        st.subheader("🔍 Detailed Analysis")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "🚨 Anomalies",
            "📊 Statistics", 
            "🕐 Seasonal",
            "💡 Insights"
        ])
        
        with tab1:
            self._render_anomaly_detection(analysis)
        
        with tab2:
            self._render_statistical_summary(analysis)
        
        with tab3:
            self._render_seasonal_analysis(analysis)
        
        with tab4:
            self._render_insights_and_recommendations(analysis)
    
    def _render_anomaly_detection(self, analysis: Dict[str, Any]) -> None:
        """Render anomaly detection results"""
        anomalies = analysis.get('anomaly_detection', {})
        
        if not anomalies:
            st.info("No anomaly detection data available")
            return
        
        # Low activity periods
        if 'invisible_work_periods' in anomalies and anomalies['invisible_work_periods']:
            st.write("**🚨 Potential 'Invisible Work' Periods:**")
            st.write("Months with high meetings but low email correspondence")
            for period in anomalies['invisible_work_periods']:
                st.write(f"• {period}")
        
        # Activity outliers
        if 'low_activity_months' in anomalies and anomalies['low_activity_months']:
            st.write("**📉 Unusually Low Activity Months:**")
            for month, value in anomalies['low_activity_months'].items():
                st.write(f"• {month}: {value:.0f} activities")
        
        # Sudden drops
        drop_keys = [key for key in anomalies.keys() if 'sudden_drops' in key]
        if drop_keys:
            st.write("**⬇️ Sudden Activity Drops (>50% decrease):**")
            for key in drop_keys:
                drops = anomalies[key]
                if drops:
                    activity_type = key.replace('_sudden_drops', '').replace('_', ' ')
                    st.write(f"**{activity_type}:**")
                    for month, change in drops.items():
                        st.write(f"• {month}: {change:.1%} decrease")
    
    def _render_statistical_summary(self, analysis: Dict[str, Any]) -> None:
        """Render statistical summary"""
        monthly_trends = analysis.get('monthly_trends', pd.DataFrame())
        pattern_analysis = analysis.get('pattern_analysis', {})
        
        if monthly_trends.empty:
            st.info("No statistical data available")
            return
        
        # Create summary statistics table
        stats_data = []
        
        for col in ['Total_Activity', 'Unique_Meetings', 'Unique_Emails', 'Correlation_Ratio']:
            if col in monthly_trends.columns:
                series = monthly_trends[col]
                stats_data.append({
                    'Metric': col.replace('_', ' '),
                    'Mean': f"{series.mean():.2f}",
                    'Median': f"{series.median():.2f}",
                    'Std Dev': f"{series.std():.2f}",
                    'Min': f"{series.min():.2f}",
                    'Max': f"{series.max():.2f}"
                })
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    def _render_seasonal_analysis(self, analysis: Dict[str, Any]) -> None:
        """Render seasonal pattern analysis"""
        seasonal = analysis.get('seasonal_analysis', {})
        
        if not seasonal:
            st.info("No seasonal analysis data available")
            return
        
        # Quarterly patterns
        if 'quarterly_trends' in seasonal:
            quarterly = seasonal['quarterly_trends']
            if not quarterly.empty:
                st.write("**📅 Quarterly Activity Patterns:**")
                st.dataframe(quarterly, use_container_width=True)
        
        # Hourly patterns
        if 'hourly_patterns' in seasonal:
            hourly = seasonal['hourly_patterns']
            if not hourly.empty:
                st.write("**🕐 Hourly Activity Patterns:**")
                
                fig = px.line(
                    hourly.reset_index(),
                    x='Hour',
                    y=hourly.columns.tolist(),
                    title="Activity by Hour of Day",
                    color_discrete_sequence=self.config.COLOR_SCHEMES['primary']
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_insights_and_recommendations(self, analysis: Dict[str, Any]) -> None:
        """Render insights and recommendations"""
        st.write("### 💡 Key Insights")
        
        pattern_analysis = analysis.get('pattern_analysis', {})
        anomaly_detection = analysis.get('anomaly_detection', {})
        correlation_analysis = analysis.get('correlation_analysis', {})
        
        insights = []
        
        # Trend insights
        activity_trend = pattern_analysis.get('activity_trend', 'Unknown')
        if activity_trend == 'Decreasing':
            insights.append("📉 **Declining Activity**: Overall activity levels are decreasing over time")
        elif activity_trend == 'Increasing':
            insights.append("📈 **Growing Activity**: Activity levels are increasing over time")
        
        # Correlation insights
        avg_correlation = correlation_analysis.get('avg_correlation_ratio', 0)
        if avg_correlation < 1:
            insights.append(f"📧 **Low Email Engagement**: Average of {avg_correlation:.2f} emails per meeting suggests limited follow-up")
        elif avg_correlation > 3:
            insights.append(f"📧 **High Email Activity**: Average of {avg_correlation:.2f} emails per meeting shows strong communication")
        
        # Anomaly insights
        invisible_work = anomaly_detection.get('invisible_work_periods', [])
        if invisible_work:
            insights.append(f"⚠️ **Invisible Work Detected**: {len(invisible_work)} periods with high meetings but low emails")
        
        for insight in insights:
            st.write(insight)
        
        # Recommendations
        st.write("### 🎯 Recommendations")
        recommendations = self._generate_recommendations(analysis)
        for rec in recommendations:
            st.write(f"• {rec}")
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        pattern_analysis = analysis.get('pattern_analysis', {})
        anomaly_detection = analysis.get('anomaly_detection', {})
        correlation_analysis = analysis.get('correlation_analysis', {})
        
        # Activity trend recommendations
        activity_trend = pattern_analysis.get('activity_trend', 'Unknown')
        if activity_trend == 'Decreasing':
            recommendations.extend([
                "Investigate causes of declining activity levels",
                "Implement strategies to maintain consistent engagement",
                "Review workload distribution and resource allocation"
            ])
        
        # Correlation recommendations
        avg_correlation = correlation_analysis.get('avg_correlation_ratio', 0)
        if avg_correlation < 1:
            recommendations.extend([
                "Improve email follow-up protocols after meetings",
                "Ensure action items from meetings are documented and shared"
            ])
        
        # Anomaly recommendations
        invisible_work = anomaly_detection.get('invisible_work_periods', [])
        if invisible_work:
            recommendations.append("Review periods of high meetings/low emails for documentation gaps")
        
        # General recommendations
        recommendations.extend([
            "Monitor monthly activity trends for early warning of engagement issues",
            "Establish consistent monthly activity targets",
            "Regular review of correlation patterns to optimize communication efficiency"
        ])
        
        return recommendations
    
    def get_report_summary(self) -> Dict[str, Any]:
        """Get summary metrics for dashboard overview"""
        analysis = self.generate_analysis()
        pattern_analysis = analysis.get('pattern_analysis', {})
        data_range = analysis.get('data_range', {})
        
        return {
            'analysis_period_months': data_range.get('total_months', 0),
            'activity_trend': pattern_analysis.get('activity_trend', 'Unknown'),
            'avg_monthly_activity': pattern_analysis.get('avg_monthly_activity', 0),
            'avg_correlation_ratio': pattern_analysis.get('avg_correlation_ratio', 0)
        }