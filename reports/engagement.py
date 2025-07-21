"""
Meeting Engagement Validation Report
Identifies meetings that had no prior or post-email correspondence within the 48-hour window
"""

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List
from datetime import datetime

from .base import BaseReport
from logger import log_performance


class EngagementReport(BaseReport):
    """
    Report 1: Meeting Engagement Validation
    
    Purpose: Identify meetings that had no prior or post-email correspondence 
    within the 48-hour window, suggesting lack of preparation or follow-up.
    
    Output: List of meetings with zero correlated emails
    Value: Exposes superficial or non-engaged participation
    """
    
    def get_required_columns(self) -> List[str]:
        """Get required columns for engagement analysis"""
        return [
            'Meeting Subject',
            'Meeting Time (BST/GMT)',
            'Email Subject',
            'Time Delta (hrs)'
        ]
    
    @log_performance
    def generate_analysis(self) -> Dict[str, Any]:
        """
        Generate meeting engagement analysis
        
        Returns:
            Dict[str, Any]: Analysis results
        """
        self.log_method_entry("generate_analysis")
        
        # Get all unique meetings
        meetings_all = self.data[['Meeting Subject', 'Meeting Time (BST/GMT)']].drop_duplicates()
        
        self.logger.info("Found %d unique meetings", len(meetings_all))
        
        # Count emails per meeting
        engaged_meetings = (
            self.data.groupby(['Meeting Subject', 'Meeting Time (BST/GMT)'])
            .size()
            .reset_index(name='Email Count')
        )
        
        # Merge to get all meetings with email counts
        all_meetings = meetings_all.merge(
            engaged_meetings, 
            on=['Meeting Subject', 'Meeting Time (BST/GMT)'], 
            how='left'
        ).fillna(0)
        
        # Identify meetings with no engagement
        no_engagement_meetings = all_meetings[all_meetings['Email Count'] == 0]
        
        # Calculate engagement statistics
        total_meetings = len(all_meetings)
        engaged_meeting_count = len(all_meetings[all_meetings['Email Count'] > 0])
        no_engagement_count = len(no_engagement_meetings)
        
        engagement_rate = (engaged_meeting_count / total_meetings) * 100 if total_meetings > 0 else 0
        
        # Additional analysis
        email_distribution = all_meetings['Email Count'].value_counts().sort_index()
        
        # Get meetings with different engagement levels
        high_engagement = all_meetings[all_meetings['Email Count'] >= 5]
        medium_engagement = all_meetings[
            (all_meetings['Email Count'] >= 2) & (all_meetings['Email Count'] < 5)
        ]
        low_engagement = all_meetings[
            (all_meetings['Email Count'] == 1)
        ]
        
        results = {
            'total_meetings': total_meetings,
            'engaged_meetings': engaged_meeting_count,
            'no_engagement_meetings': no_engagement_count,
            'engagement_rate': engagement_rate,
            'no_engagement_rate': (no_engagement_count / total_meetings) * 100 if total_meetings > 0 else 0,
            'high_engagement_count': len(high_engagement),
            'medium_engagement_count': len(medium_engagement),
            'low_engagement_count': len(low_engagement),
            'email_distribution': email_distribution,
            'no_engagement_details': no_engagement_meetings,
            'all_meetings_with_counts': all_meetings,
            'high_engagement_details': high_engagement,
            'medium_engagement_details': medium_engagement,
            'low_engagement_details': low_engagement
        }
        
        self.logger.info("Engagement analysis completed: %d/%d meetings have email correspondence (%.1f%%)",
                        engaged_meeting_count, total_meetings, engagement_rate)
        
        self.log_method_exit("generate_analysis", results)
        return results
    
    def render_report(self) -> None:
        """Render the complete engagement report"""
        self.log_method_entry("render_report")
        
        # Generate analysis
        analysis = self.generate_analysis()
        
        # Render header
        self.render_header(
            "Meeting Engagement Validation",
            "Identifies meetings with no email correspondence within the 48-hour window",
            "🎯"
        )
        
        # Render key metrics
        self._render_engagement_metrics(analysis)
        
        # Render visualizations
        self._render_engagement_charts(analysis)
        
        # Render detailed findings
        self._render_detailed_findings(analysis)
        
        self.log_method_exit("render_report")
    
    def _render_engagement_metrics(self, analysis: Dict[str, Any]) -> None:
        """Render key engagement metrics"""
        st.subheader("📊 Key Metrics")
        
        metrics = {
            "Total Meetings": f"{analysis['total_meetings']:,}",
            "Engaged Meetings": f"{analysis['engaged_meetings']:,}",
            "No Email Meetings": f"{analysis['no_engagement_meetings']:,}",
            "Engagement Rate": f"{analysis['engagement_rate']:.1f}%"
        }
        
        self.render_metrics_grid(metrics, columns=4)
        
        # Add interpretation
        if analysis['engagement_rate'] < 50:
            st.error("🚨 **Low Engagement Alert**: Less than 50% of meetings have email correspondence")
        elif analysis['engagement_rate'] < 75:
            st.warning("⚠️ **Moderate Engagement**: Room for improvement in meeting preparation/follow-up")
        else:
            st.success("✅ **Good Engagement**: Most meetings have email correspondence")
    
    def _render_engagement_charts(self, analysis: Dict[str, Any]) -> None:
        """Render engagement visualization charts"""
        st.subheader("📈 Engagement Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Engagement overview pie chart
            engagement_data = pd.DataFrame({
                'Category': ['No Emails', 'Low (1 email)', 'Medium (2-4 emails)', 'High (5+ emails)'],
                'Count': [
                    analysis['no_engagement_meetings'],
                    analysis['low_engagement_count'],
                    analysis['medium_engagement_count'],
                    analysis['high_engagement_count']
                ]
            })
            
            fig_pie = px.pie(
                engagement_data,
                names='Category',
                values='Count',
                title="Meeting Engagement Distribution",
                color_discrete_sequence=self.config.COLOR_SCHEMES['engagement']
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Email count distribution
            if len(analysis['email_distribution']) > 0:
                dist_df = pd.DataFrame({
                    'Email Count': analysis['email_distribution'].index,
                    'Meeting Count': analysis['email_distribution'].values
                })
                
                fig_bar = px.bar(
                    dist_df,
                    x='Email Count',
                    y='Meeting Count',
                    title="Distribution of Email Counts per Meeting",
                    color_discrete_sequence=self.config.COLOR_SCHEMES['primary']
                )
                fig_bar.update_layout(xaxis_title="Number of Emails", yaxis_title="Number of Meetings")
                st.plotly_chart(fig_bar, use_container_width=True)
    
    def _render_detailed_findings(self, analysis: Dict[str, Any]) -> None:
        """Render detailed findings and tables"""
        st.subheader("🔍 Detailed Findings")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "❌ No Engagement", 
            "⭐ High Engagement", 
            "📊 All Meetings",
            "💡 Insights"
        ])
        
        with tab1:
            self._render_no_engagement_table(analysis)
        
        with tab2:
            self._render_high_engagement_table(analysis)
        
        with tab3:
            self._render_all_meetings_table(analysis)
        
        with tab4:
            self._render_insights(analysis)
    
    def _render_no_engagement_table(self, analysis: Dict[str, Any]) -> None:
        """Render table of meetings with no engagement"""
        no_engagement_df = analysis['no_engagement_details']
        
        if len(no_engagement_df) == 0:
            st.success("🎉 Excellent! All meetings have some email correspondence.")
            return
        
        st.write(f"**{len(no_engagement_df)} meetings found with no email correspondence:**")
        
        # Format the display
        display_df = no_engagement_df.copy()
        if 'Meeting Time (BST/GMT)' in display_df.columns:
            display_df['Meeting Time (BST/GMT)'] = pd.to_datetime(
                display_df['Meeting Time (BST/GMT)']
            ).dt.strftime('%Y-%m-%d %H:%M')
        
        st.dataframe(
            display_df[['Meeting Subject', 'Meeting Time (BST/GMT)']],
            use_container_width=True,
            hide_index=True
        )
        
        # Risk assessment
        risk_level = self._assess_engagement_risk(len(no_engagement_df), analysis['total_meetings'])
        st.write(f"**Risk Assessment**: {risk_level}")
    
    def _render_high_engagement_table(self, analysis: Dict[str, Any]) -> None:
        """Render table of meetings with high engagement"""
        high_engagement_df = analysis['high_engagement_details']
        
        if len(high_engagement_df) == 0:
            st.info("No meetings found with high engagement (5+ emails)")
            return
        
        st.write(f"**{len(high_engagement_df)} meetings with high engagement (5+ emails):**")
        
        # Format the display
        display_df = high_engagement_df.copy()
        if 'Meeting Time (BST/GMT)' in display_df.columns:
            display_df['Meeting Time (BST/GMT)'] = pd.to_datetime(
                display_df['Meeting Time (BST/GMT)']
            ).dt.strftime('%Y-%m-%d %H:%M')
        
        st.dataframe(
            display_df[['Meeting Subject', 'Meeting Time (BST/GMT)', 'Email Count']],
            use_container_width=True,
            hide_index=True
        )
    
    def _render_all_meetings_table(self, analysis: Dict[str, Any]) -> None:
        """Render table of all meetings with email counts"""
        all_meetings_df = analysis['all_meetings_with_counts']
        
        # Format the display
        display_df = all_meetings_df.copy()
        if 'Meeting Time (BST/GMT)' in display_df.columns:
            display_df['Meeting Time (BST/GMT)'] = pd.to_datetime(
                display_df['Meeting Time (BST/GMT)']
            ).dt.strftime('%Y-%m-%d %H:%M')
        
        # Add engagement category
        def categorize_engagement(count):
            if count == 0:
                return "❌ No Emails"
            elif count == 1:
                return "🔸 Low (1 email)"
            elif count < 5:
                return "🔶 Medium (2-4 emails)"
            else:
                return "⭐ High (5+ emails)"
        
        display_df['Engagement Level'] = display_df['Email Count'].apply(categorize_engagement)
        
        # Sort by email count descending
        display_df = display_df.sort_values('Email Count', ascending=False)
        
        st.dataframe(
            display_df[['Meeting Subject', 'Meeting Time (BST/GMT)', 'Email Count', 'Engagement Level']],
            use_container_width=True,
            hide_index=True
        )
    
    def _render_insights(self, analysis: Dict[str, Any]) -> None:
        """Render analytical insights"""
        st.write("### 💡 Key Insights")
        
        insights = []
        
        # Engagement rate insight
        engagement_rate = analysis['engagement_rate']
        if engagement_rate < 60:
            insights.append(f"🔴 **Low Engagement Warning**: Only {engagement_rate:.1f}% of meetings have email correspondence")
        elif engagement_rate >= 90:
            insights.append(f"🟢 **Excellent Engagement**: {engagement_rate:.1f}% of meetings have email correspondence")
        else:
            insights.append(f"🟡 **Moderate Engagement**: {engagement_rate:.1f}% of meetings have email correspondence")
        
        # High engagement insight
        high_pct = (analysis['high_engagement_count'] / analysis['total_meetings']) * 100
        insights.append(f"📈 **High Engagement Rate**: {high_pct:.1f}% of meetings have 5+ emails")
        
        # No engagement insight
        no_eng_pct = analysis['no_engagement_rate']
        if no_eng_pct > 25:
            insights.append(f"⚠️ **Significant Gap**: {no_eng_pct:.1f}% of meetings have zero email correspondence")
        
        # Email distribution insight
        if len(analysis['email_distribution']) > 0:
            max_emails = analysis['email_distribution'].index.max()
            avg_emails = (analysis['all_meetings_with_counts']['Email Count']).mean()
            insights.append(f"📊 **Email Distribution**: Maximum {max_emails} emails per meeting, average {avg_emails:.1f}")
        
        for insight in insights:
            st.write(insight)
        
        # Recommendations
        st.write("### 🎯 Recommendations")
        recommendations = self._generate_recommendations(analysis)
        for rec in recommendations:
            st.write(f"• {rec}")
    
    def _assess_engagement_risk(self, no_engagement_count: int, total_meetings: int) -> str:
        """Assess risk level based on engagement metrics"""
        if total_meetings == 0:
            return "Unable to assess - no meetings found"
        
        no_engagement_rate = (no_engagement_count / total_meetings) * 100
        
        if no_engagement_rate > 40:
            return "🔴 **High Risk** - Significant number of meetings lack email engagement"
        elif no_engagement_rate > 20:
            return "🟡 **Medium Risk** - Moderate number of meetings lack email engagement"
        elif no_engagement_rate > 10:
            return "🟢 **Low Risk** - Most meetings have email engagement"
        else:
            return "✅ **Very Low Risk** - Excellent email engagement across meetings"
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        no_eng_rate = analysis['no_engagement_rate']
        
        if no_eng_rate > 30:
            recommendations.extend([
                "Implement mandatory pre-meeting email preparation requirements",
                "Establish post-meeting follow-up protocols within 24 hours",
                "Review meeting necessity - some may be redundant or ineffective"
            ])
        elif no_eng_rate > 15:
            recommendations.extend([
                "Improve meeting preparation by requiring agenda circulation via email",
                "Encourage post-meeting action item distribution"
            ])
        
        if analysis['high_engagement_count'] > 0:
            recommendations.append("Analyze high-engagement meetings as best practice examples")
        
        recommendations.append("Monitor engagement trends monthly to identify patterns")
        
        return recommendations
    
    def get_report_summary(self) -> Dict[str, Any]:
        """Get summary metrics for dashboard overview"""
        analysis = self.generate_analysis()
        
        return {
            'total_meetings': analysis['total_meetings'],
            'engagement_rate': analysis['engagement_rate'],
            'no_engagement_meetings': analysis['no_engagement_meetings'],
            'risk_level': self._assess_engagement_risk(
                analysis['no_engagement_meetings'], 
                analysis['total_meetings']
            )
        }