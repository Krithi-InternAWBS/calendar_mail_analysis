"""
Dashboard Layout Manager
Handles Streamlit UI components, navigation, and page routing
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime

from logger import get_logger, LoggerMixin
from config import DashboardConfig

# Import reports directly
from reports.engagement import EngagementReport
from reports.responsiveness import ResponsivenessReport
from reports.client_mapping import ClientMappingReport
from reports.content_quality import ContentQualityReport
from reports.trends import TrendsReport


class DashboardLayout(LoggerMixin):
    """
    Manages the overall dashboard layout, navigation, and page routing
    """
    
    def __init__(self):
        """Initialize the dashboard layout manager"""
        self.config = DashboardConfig()
        self.logger.info("DashboardLayout initialized")
        
        # Define page configuration
        self.pages = {
            "Home": {
                "title": "🏠 Home",
                "icon": "🏠",
                "description": "Data loading and overview"
            },
            "Report 1": {
                "title": "🎯 Meeting Engagement",
                "icon": "🎯", 
                "description": "Meetings with zero email correspondence",
                "report_class": EngagementReport
            },
            "Report 2": {
                "title": "⚡ Responsiveness Analysis",
                "icon": "⚡",
                "description": "Email response time analysis",
                "report_class": ResponsivenessReport
            },
            "Report 3": {
                "title": "🗺️ Client Communication",
                "icon": "🗺️",
                "description": "Client stakeholder engagement mapping",
                "report_class": ClientMappingReport
            },
            "Report 4": {
                "title": "📝 Content Quality",
                "icon": "📝",
                "description": "Email content substance analysis",
                "report_class": ContentQualityReport
            },
            "Report 5": {
                "title": "📈 Activity Trends",
                "icon": "📈",
                "description": "Monthly engagement pattern analysis",
                "report_class": TrendsReport
            }
        }
    
    def render_sidebar(self) -> str:
        """
        Render the sidebar navigation
        
        Returns:
            str: Selected page name
        """
        self.log_method_entry("render_sidebar")
        
        with st.sidebar:
            # App header
            st.markdown("""
            <div style="text-align: center; padding: 1rem 0;">
                <h2 style="color: #1f77b4; margin: 0;">📊 Excel Dashboard</h2>
                <p style="color: #666; margin: 0; font-size: 0.9rem;">Meeting-Email Analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Navigation menu
            st.markdown("### 🧭 Navigation")
            
            # Create radio buttons for navigation
            page_options = []
            page_labels = []
            
            for page_key, page_info in self.pages.items():
                page_options.append(page_key)
                page_labels.append(page_info["title"])
            
            selected_page = st.radio(
                "Select a page:",
                options=page_options,
                format_func=lambda x: self.pages[x]["title"],
                key="page_selector"
            )
            
            # Display page description
            if selected_page in self.pages:
                page_info = self.pages[selected_page]
                st.info(f"📋 {page_info['description']}")
            
            st.markdown("---")
            
            # Data status
            self._render_data_status_sidebar()
            
            # Additional information
            self._render_sidebar_footer()
        
        self.log_method_exit("render_sidebar", selected_page)
        return selected_page
    
    def _render_data_status_sidebar(self) -> None:
        """Render data loading status in sidebar"""
        st.markdown("### 📊 Data Status")
        
        if st.session_state.get('data_loaded', False):
            data = st.session_state.get('processed_data')
            if data is not None:
                st.success("✅ Data Loaded")
                st.metric("Records", f"{len(data):,}")
                
                # Quick stats
                if 'Meeting Subject' in data.columns:
                    unique_meetings = data['Meeting Subject'].nunique()
                    st.metric("Meetings", f"{unique_meetings:,}")
                
                if 'Email Subject' in data.columns:
                    unique_emails = data['Email Subject'].nunique()
                    st.metric("Emails", f"{unique_emails:,}")
            else:
                st.error("❌ Data Error")
        else:
            st.warning("⏳ No Data Loaded")
            st.info("👆 Load data from the Home page")
    
    def _render_sidebar_footer(self) -> None:
        """Render sidebar footer with app information"""
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.8rem; color: #666;">
            <p><strong>Excel Dashboard v1.0</strong></p>
            <p>Meeting-Email Correlation Analysis</p>
            <p>Built with Streamlit & Plotly</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_page(self, page_name: str, data: pd.DataFrame) -> None:
        """
        Render the selected page content
        
        Args:
            page_name: Name of the page to render
            data: Processed data for analysis
        """
        self.log_method_entry("render_page", page=page_name, data_shape=data.shape)
        
        try:
            if page_name == "Home":
                self._render_home_page(data)
            elif page_name in self.pages and "report_class" in self.pages[page_name]:
                self._render_report_page(page_name, data)
            else:
                st.error(f"Unknown page: {page_name}")
                
        except Exception as e:
            error_msg = f"Error rendering page {page_name}: {str(e)}"
            self.logger.error(error_msg)
            st.error(error_msg)
            
        self.log_method_exit("render_page")
    
    def _render_home_page(self, data: pd.DataFrame) -> None:
        """
        Render the home page with dashboard overview
        
        Args:
            data: Processed data
        """
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="color: #1f77b4; margin-bottom: 0;">📊 Meeting-Email Dashboard</h1>
            <p style="color: #666; font-size: 1.2rem;">Forensic Analysis of Communication Patterns</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Dashboard overview
        st.markdown("### 🎯 Dashboard Overview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            This dashboard provides comprehensive analysis of meeting engagement and email 
            communication patterns. It correlates Outlook calendar meetings with email 
            communications within 48-hour windows to validate engagement claims.
            
            **📋 Available Reports:**
            """)
            
            # List all reports
            for page_key, page_info in self.pages.items():
                if page_key != "Home":
                    st.markdown(f"• **{page_info['title']}**: {page_info['description']}")
        
        with col2:
            # Quick stats card
            if len(data) > 0:
                self._render_quick_stats_card(data)
        
        # Report summary grid
        if len(data) > 0:
            st.markdown("### 📈 Report Summary")
            self._render_report_summary_grid(data)
        
        # Recent activity
        st.markdown("### 📅 Recent Activity Preview")
        self._render_recent_activity_preview(data)
    
    def _render_quick_stats_card(self, data: pd.DataFrame) -> None:
        """Render quick statistics card"""
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
            <h3 style="margin: 0; color: white;">📊 Quick Stats</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate quick metrics
        total_records = len(data)
        unique_meetings = data['Meeting Subject'].nunique() if 'Meeting Subject' in data.columns else 0
        unique_emails = data['Email Subject'].nunique() if 'Email Subject' in data.columns else 0
        
        # Date range
        date_range = "N/A"
        if 'Meeting Time (BST/GMT)' in data.columns:
            meeting_times = pd.to_datetime(data['Meeting Time (BST/GMT)'], errors='coerce').dropna()
            if len(meeting_times) > 0:
                start_date = meeting_times.min().strftime('%Y-%m-%d')
                end_date = meeting_times.max().strftime('%Y-%m-%d')
                date_range = f"{start_date} to {end_date}"
        
        st.metric("Total Records", f"{total_records:,}")
        st.metric("Unique Meetings", f"{unique_meetings:,}")
        st.metric("Unique Emails", f"{unique_emails:,}")
        st.caption(f"📅 **Period**: {date_range}")
    
    def _render_report_summary_grid(self, data: pd.DataFrame) -> None:
        """Render grid of report summaries"""
        
        # Create summary cards for each report
        report_summaries = {}
        
        # Generate summaries for each report
        for page_key, page_info in self.pages.items():
            if "report_class" in page_info:
                try:
                    report_class = page_info["report_class"]
                    report_instance = report_class(data)
                    
                    if report_instance.validate_data():
                        summary = report_instance.get_report_summary()
                        report_summaries[page_key] = summary
                    else:
                        report_summaries[page_key] = {"status": "Data validation failed"}
                        
                except Exception as e:
                    self.logger.warning("Failed to generate summary for %s: %s", page_key, str(e))
                    report_summaries[page_key] = {"status": "Error generating summary"}
        
        # Display summary cards
        cols = st.columns(len(report_summaries))
        
        for i, (page_key, summary) in enumerate(report_summaries.items()):
            with cols[i % len(cols)]:
                page_info = self.pages[page_key]
                self._render_summary_card(page_info, summary)
    
    def _render_summary_card(self, page_info: Dict[str, Any], summary: Dict[str, Any]) -> None:
        """Render individual report summary card"""
        
        card_color = page_info.get("color", "#3498db")
        
        # Card header
        st.markdown(f"""
        <div style="background: {card_color}; padding: 1rem; border-radius: 8px 8px 0 0; color: white;">
            <h4 style="margin: 0; color: white;">{page_info['icon']} {page_info['title'].split()[-1]}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Card content
        with st.container():
            if "status" in summary:
                st.warning(summary["status"])
            else:
                # Display key metrics from summary
                for key, value in summary.items():
                    if isinstance(value, (int, float)):
                        if isinstance(value, float):
                            st.metric(key.replace("_", " ").title(), f"{value:.2f}")
                        else:
                            st.metric(key.replace("_", " ").title(), f"{value:,}")
                    else:
                        st.caption(f"**{key.replace('_', ' ').title()}**: {value}")
    
    def _render_recent_activity_preview(self, data: pd.DataFrame) -> None:
        """Render preview of recent activity"""
        
        if len(data) == 0:
            st.info("No data available for preview")
            return
        
        # Show recent meetings and emails
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📅 Recent Meetings")
            
            if 'Meeting Time (BST/GMT)' in data.columns and 'Meeting Subject' in data.columns:
                # Get recent meetings
                meeting_data = data[['Meeting Time (BST/GMT)', 'Meeting Subject']].drop_duplicates()
                meeting_data['Meeting Time (BST/GMT)'] = pd.to_datetime(
                    meeting_data['Meeting Time (BST/GMT)'], errors='coerce'
                )
                meeting_data = meeting_data.dropna().sort_values(
                    'Meeting Time (BST/GMT)', ascending=False
                ).head(5)
                
                if len(meeting_data) > 0:
                    for _, row in meeting_data.iterrows():
                        date_str = row['Meeting Time (BST/GMT)'].strftime('%Y-%m-%d %H:%M')
                        st.write(f"• **{date_str}**: {row['Meeting Subject'][:50]}...")
                else:
                    st.info("No recent meetings found")
            else:
                st.info("Meeting data not available")
        
        with col2:
            st.markdown("#### 📧 Recent Emails")
            
            if 'Email Time (BST/GMT)' in data.columns and 'Email Subject' in data.columns:
                # Get recent emails
                email_data = data[['Email Time (BST/GMT)', 'Email Subject', 'Direction']].drop_duplicates()
                email_data['Email Time (BST/GMT)'] = pd.to_datetime(
                    email_data['Email Time (BST/GMT)'], errors='coerce'
                )
                email_data = email_data.dropna().sort_values(
                    'Email Time (BST/GMT)', ascending=False
                ).head(5)
                
                if len(email_data) > 0:
                    for _, row in email_data.iterrows():
                        date_str = row['Email Time (BST/GMT)'].strftime('%Y-%m-%d %H:%M')
                        direction = row.get('Direction', 'Unknown')
                        direction_icon = "📤" if direction == "Sent" else "📥" if direction == "Received" else "📧"
                        st.write(f"• {direction_icon} **{date_str}**: {row['Email Subject'][:50]}...")
                else:
                    st.info("No recent emails found")
            else:
                st.info("Email data not available")
    
    def _render_report_page(self, page_name: str, data: pd.DataFrame) -> None:
        """
        Render a specific report page
        
        Args:
            page_name: Name of the report page
            data: Processed data for analysis
        """
        page_info = self.pages[page_name]
        report_class = page_info["report_class"]
        
        try:
            # Create report instance
            report_instance = report_class(data)
            
            # Add data filtering options
            filtered_data = self._render_data_filters(data)
            
            # Update report data if filtered
            if len(filtered_data) != len(data):
                report_instance = report_class(filtered_data)
                st.info(f"📊 **Filtered Data**: Showing {len(filtered_data):,} of {len(data):,} total records")
            
            # Run the report
            report_instance.run_report()
            
        except Exception as e:
            error_msg = f"Error running report {page_name}: {str(e)}"
            self.logger.error(error_msg)
            st.error(error_msg)
            st.exception(e)
    
    def _render_data_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Render data filtering options
        
        Args:
            data: Original data
            
        Returns:
            pd.DataFrame: Filtered data
        """
        with st.expander("🔍 Data Filters", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            filtered_data = data.copy()
            
            with col1:
                # Time range filter
                if 'Meeting Time (BST/GMT)' in data.columns:
                    st.markdown("**📅 Time Range**")
                    
                    time_options = [
                        "All Data",
                        "Last 30 days", 
                        "Last 90 days",
                        "Last 6 months",
                        "Last year"
                    ]
                    
                    selected_timeframe = st.selectbox(
                        "Select timeframe:",
                        options=time_options,
                        key="timeframe_filter"
                    )
                    
                    if selected_timeframe != "All Data":
                        filtered_data = self._apply_time_filter(filtered_data, selected_timeframe)
            
            with col2:
                # Direction filter
                if 'Direction' in data.columns:
                    st.markdown("**↔️ Email Direction**")
                    
                    directions = ["All"] + list(data['Direction'].dropna().unique())
                    selected_direction = st.selectbox(
                        "Filter by direction:",
                        options=directions,
                        key="direction_filter"
                    )
                    
                    if selected_direction != "All":
                        filtered_data = filtered_data[filtered_data['Direction'] == selected_direction]
            
            with col3:
                # Domain filter
                if 'Sender Domain' in data.columns or 'Receiver Domain' in data.columns:
                    st.markdown("**🌐 Domain Filter**")
                    
                    # Get unique domains
                    domains = set()
                    for col in ['Sender Domain', 'Receiver Domain']:
                        if col in data.columns:
                            domains.update(data[col].dropna().unique())
                    
                    domains = ["All"] + sorted(list(domains))
                    selected_domain = st.selectbox(
                        "Filter by domain:",
                        options=domains,
                        key="domain_filter"
                    )
                    
                    if selected_domain != "All":
                        domain_mask = (
                            (data.get('Sender Domain', '') == selected_domain) |
                            (data.get('Receiver Domain', '') == selected_domain)
                        )
                        filtered_data = filtered_data[domain_mask]
            
            # Show filter summary
            if len(filtered_data) != len(data):
                reduction_pct = ((len(data) - len(filtered_data)) / len(data)) * 100
                st.info(f"🔽 **Filter Applied**: {reduction_pct:.1f}% of data filtered out")
        
        return filtered_data
    
    def _apply_time_filter(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Apply time-based filtering to data
        
        Args:
            data: Original data
            timeframe: Selected timeframe
            
        Returns:
            pd.DataFrame: Time-filtered data
        """
        if 'Meeting Time (BST/GMT)' not in data.columns:
            return data
        
        # Convert to datetime
        meeting_times = pd.to_datetime(data['Meeting Time (BST/GMT)'], errors='coerce')
        
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
            return data
        
        # Apply filter
        time_mask = meeting_times >= cutoff
        return data[time_mask]
    
    def render_loading_screen(self) -> None:
        """Render loading screen while processing data"""
        st.markdown("""
        <div style="text-align: center; padding: 3rem 0;">
            <h2>⏳ Processing Data...</h2>
            <p>Please wait while we analyze your meeting and email data.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress bar
        progress_bar = st.progress(0)
        
        # Simulate progress (in real implementation, this would be actual progress)
        import time
        for i in range(101):
            progress_bar.progress(i)
            time.sleep(0.01)
        
        st.success("✅ Data processing completed!")
    
    def render_error_page(self, error_message: str) -> None:
        """
        Render error page with troubleshooting information
        
        Args:
            error_message: Error message to display
        """
        st.error("❌ **Application Error**")
        st.write(f"**Error Details**: {error_message}")
        
        st.markdown("""
        ### 🔧 Troubleshooting Steps:
        
        1. **Check Data Format**: Ensure your Excel file contains the required columns
        2. **Verify File Size**: Large files may take longer to process
        3. **Refresh Application**: Try reloading the page
        4. **Contact Support**: If the issue persists, please contact the administrator
        
        ### 📋 Required Columns:
        - Meeting Subject
        - Organizer
        - Attendees
        - Email Subject
        - Mail Sender
        - Mail Receiver
        - Email Body Content
        - Direction
        - Email Time (BST/GMT)
        - Meeting Time (BST/GMT)
        - Time Delta (hrs)
        """)
        
        if st.button("🔄 Restart Application"):
            st.rerun()