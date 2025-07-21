"""
Responsiveness & Follow-Up Timeliness Analysis Report
Measures response delays for inbound/outbound emails related to meetings
"""

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from typing import Dict, Any, List
from datetime import datetime, timedelta

from .base import BaseReport
from logger import log_performance


class ResponsivenessReport(BaseReport):
    """
    Report 2: Responsiveness & Follow-Up Timeliness Analysis
    
    Purpose: Measure response delays for inbound/outbound emails related to meetings.
    Output: Average response time, % responded within 12/24/48 hrs
    Value: Helps validate Katie's claim of being responsive and collaborative
    """
    
    def get_required_columns(self) -> List[str]:
        """Get required columns for responsiveness analysis"""
        return [
            'Email Time (BST/GMT)',
            'Mail Receiver',
            'Mail Sender',
            'Direction',
            'Email Subject',
            'Meeting Subject'
        ]
    
    @log_performance
    def generate_analysis(self) -> Dict[str, Any]:
        """
        Generate responsiveness analysis
        
        Returns:
            Dict[str, Any]: Analysis results
        """
        self.log_method_entry("generate_analysis")
        
        # Sort data by receiver and email time for response calculation
        df_sorted = self.data.sort_values(['Mail Receiver', 'Email Time (BST/GMT)'])
        
        # Calculate response times
        response_analysis = self._calculate_response_times(df_sorted)
        
        # Calculate response rate statistics
        response_stats = self._calculate_response_statistics(response_analysis)
        
        # Analyze response patterns
        pattern_analysis = self._analyze_response_patterns(response_analysis)
        
        # Direction-based analysis
        direction_analysis = self._analyze_by_direction(response_analysis)
        
        results = {
            'response_data': response_analysis,
            'response_statistics': response_stats,
            'pattern_analysis': pattern_analysis,
            'direction_analysis': direction_analysis,
            'total_emails': len(self.data),
            'emails_with_responses': len(response_analysis[response_analysis['Response Time (hrs)'].notna()])
        }
        
        self.logger.info("Responsiveness analysis completed: %d emails analyzed, %d with calculable response times",
                        len(self.data), results['emails_with_responses'])
        
        self.log_method_exit("generate_analysis", results)
        return results
    
    def _calculate_response_times(self, df_sorted: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate response times between consecutive emails
        
        Args:
            df_sorted: DataFrame sorted by receiver and time
            
        Returns:
            pd.DataFrame: DataFrame with response times
        """
        self.log_method_entry("_calculate_response_times", shape=df_sorted.shape)
        
        # Create copy for modifications
        df_analysis = df_sorted.copy()
        
        # Group by email thread (using receiver as thread identifier)
        df_analysis['Next Email Time'] = df_analysis.groupby('Mail Receiver')['Email Time (BST/GMT)'].shift(-1)
        df_analysis['Next Email Sender'] = df_analysis.groupby('Mail Receiver')['Mail Sender'].shift(-1)
        
        # Calculate response time in hours
        df_analysis['Response Time (hrs)'] = (
            (df_analysis['Next Email Time'] - df_analysis['Email Time (BST/GMT)']).dt.total_seconds() / 3600
        )
        
        # Only keep responses where sender changes (actual responses, not same person)
        df_analysis['Is_Response'] = (
            (df_analysis['Mail Sender'] != df_analysis['Next Email Sender']) &
            (df_analysis['Response Time (hrs)'].notna()) &
            (df_analysis['Response Time (hrs)'] > 0) &
            (df_analysis['Response Time (hrs)'] <= 168)  # Within 7 days
        )
        
        # Filter to only actual responses
        response_df = df_analysis[df_analysis['Is_Response']].copy()
        
        self.logger.info("Calculated response times for %d email exchanges", len(response_df))
        self.log_method_exit("_calculate_response_times", response_df)
        
        return response_df
    
    def _calculate_response_statistics(self, response_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate response time statistics
        
        Args:
            response_df: DataFrame with response times
            
        Returns:
            Dict[str, Any]: Response statistics
        """
        if len(response_df) == 0:
            return {
                'avg_response_time': 0,
                'median_response_time': 0,
                'response_within_1hr': 0,
                'response_within_4hr': 0,
                'response_within_12hr': 0,
                'response_within_24hr': 0,
                'response_within_48hr': 0,
                'total_responses': 0
            }
        
        response_times = response_df['Response Time (hrs)'].dropna()
        
        if len(response_times) == 0:
            return self._get_empty_stats()
        
        stats = {
            'avg_response_time': response_times.mean(),
            'median_response_time': response_times.median(),
            'total_responses': len(response_times),
            'min_response_time': response_times.min(),
            'max_response_time': response_times.max(),
            'std_response_time': response_times.std()
        }
        
        # Calculate response rate categories
        for category, threshold in self.config.RESPONSE_TIME_CATEGORIES.items():
            if threshold != float('inf'):
                pct = (response_times <= threshold).mean() * 100
                stats[f'response_within_{category}'] = pct
            else:
                pct = (response_times > 48).mean() * 100
                stats[f'response_{category}'] = pct
        
        self.logger.info("Response statistics calculated: avg %.1f hrs, median %.1f hrs", 
                        stats['avg_response_time'], stats['median_response_time'])
        
        return stats
    
    def _get_empty_stats(self) -> Dict[str, Any]:
        """Return empty statistics when no data available"""
        return {
            'avg_response_time': 0,
            'median_response_time': 0,
            'response_within_immediate': 0,
            'response_within_quick': 0,
            'response_within_same_day': 0,
            'response_within_next_day': 0,
            'response_within_two_days': 0,
            'response_slow': 0,
            'total_responses': 0,
            'min_response_time': 0,
            'max_response_time': 0,
            'std_response_time': 0
        }
    
    def _analyze_response_patterns(self, response_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze response patterns by time and day
        
        Args:
            response_df: DataFrame with response times
            
        Returns:
            Dict[str, Any]: Pattern analysis results
        """
        if len(response_df) == 0:
            return {}
        
        # Convert email time to datetime for analysis
        response_df = response_df.copy()
        response_df['Email Hour'] = pd.to_datetime(response_df['Email Time (BST/GMT)']).dt.hour
        response_df['Email Day'] = pd.to_datetime(response_df['Email Time (BST/GMT)']).dt.day_name()
        response_df['Email Month'] = pd.to_datetime(response_df['Email Time (BST/GMT)']).dt.month
        
        patterns = {
            'hourly_avg_response': response_df.groupby('Email Hour')['Response Time (hrs)'].mean(),
            'daily_avg_response': response_df.groupby('Email Day')['Response Time (hrs)'].mean(),
            'monthly_avg_response': response_df.groupby('Email Month')['Response Time (hrs)'].mean(),
            'hourly_response_count': response_df.groupby('Email Hour')['Response Time (hrs)'].count(),
            'daily_response_count': response_df.groupby('Email Day')['Response Time (hrs)'].count()
        }
        
        # Find peak response hours
        if len(patterns['hourly_response_count']) > 0:
            patterns['peak_response_hour'] = patterns['hourly_response_count'].idxmax()
            patterns['fastest_response_hour'] = patterns['hourly_avg_response'].idxmin()
        
        return patterns
    
    def _analyze_by_direction(self, response_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze response times by email direction
        
        Args:
            response_df: DataFrame with response times
            
        Returns:
            Dict[str, Any]: Direction-based analysis
        """
        if len(response_df) == 0 or 'Direction' not in response_df.columns:
            return {}
        
        direction_stats = {}
        
        for direction in response_df['Direction'].unique():
            if pd.notna(direction):
                direction_data = response_df[response_df['Direction'] == direction]
                if len(direction_data) > 0:
                    direction_stats[direction] = {
                        'count': len(direction_data),
                        'avg_response_time': direction_data['Response Time (hrs)'].mean(),
                        'median_response_time': direction_data['Response Time (hrs)'].median(),
                        'within_24hr_pct': (direction_data['Response Time (hrs)'] <= 24).mean() * 100
                    }
        
        return direction_stats
    
    def render_report(self) -> None:
        """Render the complete responsiveness report"""
        self.log_method_entry("render_report")
        
        # Generate analysis
        analysis = self.generate_analysis()
        
        # Render header
        self.render_header(
            "Responsiveness & Follow-Up Timeliness Analysis",
            "Measures email response delays and collaboration patterns",
            "⚡"
        )
        
        # Render key metrics
        self._render_responsiveness_metrics(analysis)
        
        # Render visualizations
        self._render_responsiveness_charts(analysis)
        
        # Render detailed analysis
        self._render_detailed_analysis(analysis)
        
        self.log_method_exit("render_report")
    
    def _render_responsiveness_metrics(self, analysis: Dict[str, Any]) -> None:
        """Render key responsiveness metrics"""
        st.subheader("📊 Response Time Metrics")
        
        stats = analysis['response_statistics']
        
        if stats['total_responses'] == 0:
            st.warning("⚠️ No response time data available for analysis")
            return
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Average Response Time",
                f"{stats['avg_response_time']:.1f} hrs",
                help="Mean time to respond to emails"
            )
        
        with col2:
            st.metric(
                "Median Response Time", 
                f"{stats['median_response_time']:.1f} hrs",
                help="Middle value of response times"
            )
        
        with col3:
            st.metric(
                "Within 24 Hours",
                f"{stats.get('response_within_next_day', 0):.1f}%",
                help="Percentage of emails responded to within 24 hours"
            )
        
        with col4:
            st.metric(
                "Total Analyzed",
                f"{stats['total_responses']:,}",
                help="Number of email exchanges analyzed"
            )
        
        # Response time categories
        st.subheader("⏱️ Response Time Distribution")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        categories = [
            ("≤ 1 hour", "response_within_immediate", "🟢"),
            ("≤ 4 hours", "response_within_quick", "🟡"),
            ("≤ 12 hours", "response_within_same_day", "🟠"),
            ("≤ 24 hours", "response_within_next_day", "🔴"),
            ("> 48 hours", "response_slow", "⚫")
        ]
        
        for i, (label, key, emoji) in enumerate(categories):
            with [col1, col2, col3, col4, col5][i]:
                value = stats.get(key, 0)
                st.metric(f"{emoji} {label}", f"{value:.1f}%")
    
    def _render_responsiveness_charts(self, analysis: Dict[str, Any]) -> None:
        """Render responsiveness visualization charts"""
        st.subheader("📈 Response Time Analysis")
        
        stats = analysis['response_statistics']
        
        if stats['total_responses'] == 0:
            st.info("No response time data available for visualization")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Response time distribution
            self._render_response_distribution_chart(analysis)
        
        with col2:
            # Response time trends
            self._render_response_trends_chart(analysis)
        
        # Pattern analysis charts
        if analysis.get('pattern_analysis'):
            self._render_pattern_charts(analysis)
    
    def _render_response_distribution_chart(self, analysis: Dict[str, Any]) -> None:
        """Render response time distribution chart"""
        response_df = analysis['response_data']
        
        if len(response_df) == 0:
            return
        
        # Create bins for response times
        response_times = response_df['Response Time (hrs)'].dropna()
        
        if len(response_times) > 0:
            # Create histogram
            fig = px.histogram(
                response_times,
                nbins=20,
                title="Response Time Distribution",
                labels={'value': 'Response Time (hours)', 'count': 'Number of Responses'},
                color_discrete_sequence=self.config.COLOR_SCHEMES['primary']
            )
            
            # Add vertical lines for key thresholds
            fig.add_vline(x=1, line_dash="dash", line_color="green", 
                         annotation_text="1 hour")
            fig.add_vline(x=24, line_dash="dash", line_color="orange", 
                         annotation_text="24 hours")
            fig.add_vline(x=48, line_dash="dash", line_color="red", 
                         annotation_text="48 hours")
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_response_trends_chart(self, analysis: Dict[str, Any]) -> None:
        """Render response time trends over time"""
        response_df = analysis['response_data']
        
        if len(response_df) == 0:
            return
        
        # Group by date for trend analysis
        response_df_copy = response_df.copy()
        response_df_copy['Date'] = pd.to_datetime(response_df_copy['Email Time (BST/GMT)']).dt.date
        
        daily_avg = response_df_copy.groupby('Date')['Response Time (hrs)'].mean().reset_index()
        
        if len(daily_avg) > 1:
            fig = px.line(
                daily_avg,
                x='Date',
                y='Response Time (hrs)',
                title="Average Response Time Trend",
                color_discrete_sequence=self.config.COLOR_SCHEMES['trend']
            )
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Average Response Time (hours)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_pattern_charts(self, analysis: Dict[str, Any]) -> None:
        """Render response pattern analysis charts"""
        patterns = analysis.get('pattern_analysis', {})
        
        if not patterns:
            return
        
        st.subheader("🕐 Response Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly response patterns
            if 'hourly_avg_response' in patterns and len(patterns['hourly_avg_response']) > 0:
                hourly_df = pd.DataFrame({
                    'Hour': patterns['hourly_avg_response'].index,
                    'Avg Response Time': patterns['hourly_avg_response'].values
                })
                
                fig = px.bar(
                    hourly_df,
                    x='Hour',
                    y='Avg Response Time',
                    title="Average Response Time by Hour of Day",
                    color_discrete_sequence=self.config.COLOR_SCHEMES['primary']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Daily response patterns
            if 'daily_avg_response' in patterns and len(patterns['daily_avg_response']) > 0:
                daily_df = pd.DataFrame({
                    'Day': patterns['daily_avg_response'].index,
                    'Avg Response Time': patterns['daily_avg_response'].values
                })
                
                # Reorder days
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                daily_df['Day'] = pd.Categorical(daily_df['Day'], categories=day_order, ordered=True)
                daily_df = daily_df.sort_values('Day')
                
                fig = px.bar(
                    daily_df,
                    x='Day',
                    y='Avg Response Time',
                    title="Average Response Time by Day of Week",
                    color_discrete_sequence=self.config.COLOR_SCHEMES['primary']
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_detailed_analysis(self, analysis: Dict[str, Any]) -> None:
        """Render detailed analysis and insights"""
        st.subheader("🔍 Detailed Analysis")
        
        tab1, tab2, tab3 = st.tabs(["📊 Statistics", "📧 By Direction", "💡 Insights"])
        
        with tab1:
            self._render_statistics_table(analysis)
        
        with tab2:
            self._render_direction_analysis(analysis)
        
        with tab3:
            self._render_insights(analysis)
    
    def _render_statistics_table(self, analysis: Dict[str, Any]) -> None:
        """Render detailed statistics table"""
        stats = analysis['response_statistics']
        
        if stats['total_responses'] == 0:
            st.info("No statistics available")
            return
        
        stats_df = pd.DataFrame([
            ["Total Responses Analyzed", f"{stats['total_responses']:,}"],
            ["Average Response Time", f"{stats['avg_response_time']:.2f} hours"],
            ["Median Response Time", f"{stats['median_response_time']:.2f} hours"],
            ["Fastest Response", f"{stats['min_response_time']:.2f} hours"],
            ["Slowest Response", f"{stats['max_response_time']:.2f} hours"],
            ["Standard Deviation", f"{stats['std_response_time']:.2f} hours"],
        ], columns=["Metric", "Value"])
        
        st.table(stats_df)
    
    def _render_direction_analysis(self, analysis: Dict[str, Any]) -> None:
        """Render analysis by email direction"""
        direction_stats = analysis.get('direction_analysis', {})
        
        if not direction_stats:
            st.info("No direction-based analysis available")
            return
        
        direction_data = []
        for direction, stats in direction_stats.items():
            direction_data.append({
                'Direction': direction,
                'Count': stats['count'],
                'Avg Response Time (hrs)': f"{stats['avg_response_time']:.2f}",
                'Median Response Time (hrs)': f"{stats['median_response_time']:.2f}",
                'Within 24hrs (%)': f"{stats['within_24hr_pct']:.1f}%"
            })
        
        direction_df = pd.DataFrame(direction_data)
        st.dataframe(direction_df, use_container_width=True, hide_index=True)
    
    def _render_insights(self, analysis: Dict[str, Any]) -> None:
        """Render analytical insights and recommendations"""
        stats = analysis['response_statistics']
        
        if stats['total_responses'] == 0:
            st.info("No insights available without response data")
            return
        
        st.write("### 💡 Key Insights")
        
        insights = []
        
        # Response time assessment
        avg_response = stats['avg_response_time']
        if avg_response <= 4:
            insights.append("🟢 **Excellent Responsiveness**: Average response time under 4 hours")
        elif avg_response <= 12:
            insights.append("🟡 **Good Responsiveness**: Average response time under 12 hours")
        elif avg_response <= 24:
            insights.append("🟠 **Moderate Responsiveness**: Average response time under 24 hours")
        else:
            insights.append("🔴 **Slow Responsiveness**: Average response time over 24 hours")
        
        # 24-hour response rate
        within_24hr = stats.get('response_within_next_day', 0)
        if within_24hr >= 80:
            insights.append(f"✅ **Strong 24-hour Response Rate**: {within_24hr:.1f}% of emails answered within 24 hours")
        elif within_24hr >= 60:
            insights.append(f"🟡 **Moderate 24-hour Response Rate**: {within_24hr:.1f}% of emails answered within 24 hours")
        else:
            insights.append(f"🔴 **Low 24-hour Response Rate**: Only {within_24hr:.1f}% of emails answered within 24 hours")
        
        # Pattern insights
        patterns = analysis.get('pattern_analysis', {})
        if patterns.get('peak_response_hour'):
            peak_hour = patterns['peak_response_hour']
            insights.append(f"📈 **Peak Response Time**: Most emails sent at {peak_hour}:00")
        
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
        stats = analysis['response_statistics']
        
        avg_response = stats['avg_response_time']
        within_24hr = stats.get('response_within_next_day', 0)
        
        if avg_response > 24:
            recommendations.extend([
                "Set up email response time targets (e.g., 24-hour maximum)",
                "Implement email priority classification system",
                "Consider automated acknowledgment responses"
            ])
        elif avg_response > 12:
            recommendations.extend([
                "Aim for same-day response targets",
                "Review email management workflow for efficiency gains"
            ])
        
        if within_24hr < 70:
            recommendations.append("Improve 24-hour response rate to demonstrate responsiveness")
        
        recommendations.extend([
            "Monitor response time trends monthly",
            "Document response time performance for stakeholder communication"
        ])
        
        return recommendations
    
    def get_report_summary(self) -> Dict[str, Any]:
        """Get summary metrics for dashboard overview"""
        analysis = self.generate_analysis()
        stats = analysis['response_statistics']
        
        return {
            'avg_response_time': stats['avg_response_time'],
            'within_24hr_rate': stats.get('response_within_next_day', 0),
            'total_responses': stats['total_responses'],
            'responsiveness_grade': self._calculate_responsiveness_grade(stats)
        }
    
    def _calculate_responsiveness_grade(self, stats: Dict[str, Any]) -> str:
        """Calculate overall responsiveness grade"""
        if stats['total_responses'] == 0:
            return "No Data"
        
        avg_response = stats['avg_response_time']
        within_24hr = stats.get('response_within_next_day', 0)
        
        # Weighted scoring
        time_score = max(0, 100 - (avg_response * 2))  # Penalize longer response times
        rate_score = within_24hr  # Direct percentage
        
        overall_score = (time_score * 0.6) + (rate_score * 0.4)
        
        if overall_score >= 80:
            return "A - Excellent"
        elif overall_score >= 70:
            return "B - Good"
        elif overall_score >= 60:
            return "C - Average"
        elif overall_score >= 50:
            return "D - Below Average"
        else:
            return "F - Poor"