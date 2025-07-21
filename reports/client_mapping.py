"""
Client Communication Mapping Report - Enhanced with Attendee Analysis
"""

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import re
from typing import Dict, Any, List, Set, Tuple

from .base import BaseReport
from logger import log_performance


class ClientMappingReport(BaseReport):
    """Client Communication Mapping - Shows email vs meeting engagement gaps with attendee analysis"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.info("Initializing Enhanced Client Mapping Report with attendee analysis")
        self.unwanted_chars = {')', '(', '[', ']', '{', '}', '<', '>', '"', "'", 
                              '`', '~', '!', '#', '$', '%', '^', '&', '*', '+', 
                              '=', '|', '\\', ':', ';', '?', ',', ' ', '\t', '\n', '\r'}
        self.internal_domains = {'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
                               'icloud.com', 'aol.com', 'protonmail.com', 'yandex.com'}
    
    def get_required_columns(self) -> List[str]:
        return ['Mail Receiver', 'Mail Sender', 'Direction', 'Meeting Subject', 
                'Email Subject', 'Attendees', 'Meeting Time (BST/GMT)']
    
    def _clean_domain(self, email: str) -> str:
        """Extract and clean domain from email"""
        try:
            if pd.isna(email) or '@' not in str(email):
                return 'unknown'
            
            domain = str(email).split('@')[-1]
            cleaned = ''.join(c for c in domain if c not in self.unwanted_chars).strip('.').lower()
            
            if not re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', cleaned):
                return 'unknown'
                
            return cleaned if cleaned else 'unknown'
        except Exception as e:
            self.logger.warning("Error cleaning domain for email %s: %s", email, str(e))
            return 'unknown'
    
    def _get_client_domain(self, row) -> str:
        """Determine client domain from sender/receiver"""
        sender = self._clean_domain(row['Mail Sender'])
        receiver = self._clean_domain(row['Mail Receiver'])
        
        if sender != 'unknown' and sender not in self.internal_domains:
            return sender
        elif receiver != 'unknown' and receiver not in self.internal_domains:
            return receiver
        return 'internal'
    
    def _parse_attendees(self, attendees_str: str) -> Set[str]:
        """Parse attendees string and extract individual email addresses"""
        if pd.isna(attendees_str):
            return set()
        
        try:
            # Split by common delimiters and extract emails
            delimiters = [';', ',', '\n', '\r\n', '|']
            attendees_text = str(attendees_str)
            
            for delimiter in delimiters:
                attendees_text = attendees_text.replace(delimiter, '|||')
            
            # Extract email addresses using regex
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.findall(email_pattern, attendees_text)
            
            # Clean and normalize emails
            cleaned_emails = set()
            for email in emails:
                cleaned = email.strip().lower()
                if cleaned and '@' in cleaned:
                    cleaned_emails.add(cleaned)
            
            return cleaned_emails
            
        except Exception as e:
            self.logger.warning("Error parsing attendees %s: %s", attendees_str, str(e))
            return set()
    
    def _check_email_engagement(self, attendee_email: str, email_data: pd.DataFrame) -> Dict[str, Any]:
        """Check if attendee has email engagement"""
        try:
            # Check in both sender and receiver columns
            sent_emails = email_data[
                email_data['Mail Sender'].str.lower().str.contains(attendee_email, na=False)
            ]
            received_emails = email_data[
                email_data['Mail Receiver'].str.lower().str.contains(attendee_email, na=False)
            ]
            
            total_emails = len(sent_emails) + len(received_emails)
            
            return {
                'has_email_engagement': total_emails > 0,
                'emails_sent': len(sent_emails),
                'emails_received': len(received_emails),
                'total_emails': total_emails
            }
        except Exception as e:
            self.logger.warning("Error checking email engagement for %s: %s", attendee_email, str(e))
            return {
                'has_email_engagement': False,
                'emails_sent': 0,
                'emails_received': 0,
                'total_emails': 0
            }
    
    @log_performance
    def _generate_attendee_analysis(self) -> Dict[str, Any]:
        """Generate detailed attendee engagement analysis"""
        self.logger.info("Generating attendee engagement analysis")
        
        df = self.data.copy()
        attendee_matrix = []
        
        # Get unique meetings with attendees
        meetings_data = df[df['Meeting Subject'].notna() & df['Attendees'].notna()].copy()
        unique_meetings = meetings_data.drop_duplicates(['Meeting Subject', 'Meeting Time (BST/GMT)'])
        
        self.logger.info("Processing %d unique meetings for attendee analysis", len(unique_meetings))
        
        for _, meeting_row in unique_meetings.iterrows():
            meeting_subject = meeting_row['Meeting Subject']
            meeting_time = meeting_row['Meeting Time (BST/GMT)']
            attendees = self._parse_attendees(meeting_row['Attendees'])
            
            for attendee_email in attendees:
                # Skip internal domains for client analysis
                domain = self._clean_domain(attendee_email)
                if domain in self.internal_domains or domain == 'unknown':
                    continue
                
                # Check email engagement for this attendee
                engagement = self._check_email_engagement(attendee_email, df)
                
                attendee_matrix.append({
                    'Meeting_Subject': meeting_subject,
                    'Meeting_Time': meeting_time,
                    'Attendee_Email': attendee_email,
                    'Attendee_Domain': domain,
                    'In_Meeting': True,  # All attendees are from meeting data
                    'Has_Email_Engagement': engagement['has_email_engagement'],
                    'Emails_Sent': engagement['emails_sent'],
                    'Emails_Received': engagement['emails_received'],
                    'Total_Emails': engagement['total_emails'],
                    'Engagement_Status': 'Engaged' if engagement['has_email_engagement'] else 'Not Engaged'
                })
        
        attendee_df = pd.DataFrame(attendee_matrix)
        
        if attendee_df.empty:
            self.logger.warning("No attendee data generated - check attendees column format")
            return {
                'attendee_matrix': pd.DataFrame(),
                'engagement_gaps': {},
                'summary_stats': {}
            }
        
        # Calculate summary statistics
        total_attendees = len(attendee_df)
        engaged_attendees = len(attendee_df[attendee_df['Has_Email_Engagement'] == True])
        not_engaged_attendees = total_attendees - engaged_attendees
        
        # Group by attendee for unique analysis
        unique_attendees = attendee_df.groupby('Attendee_Email').agg({
            'Has_Email_Engagement': 'any',  # True if engaged in any meeting
            'Total_Emails': 'sum',
            'Meeting_Subject': 'nunique'
        }).reset_index()
        
        unique_attendees['Attendee_Domain'] = unique_attendees['Attendee_Email'].apply(self._clean_domain)
        
        # Identify engagement gaps
        gaps = {
            'not_engaged_attendees': attendee_df[attendee_df['Has_Email_Engagement'] == False],
            'engaged_attendees': attendee_df[attendee_df['Has_Email_Engagement'] == True],
            'domains_with_gaps': attendee_df.groupby('Attendee_Domain').agg({
                'Has_Email_Engagement': ['count', 'sum']
            }).reset_index()
        }
        
        # Summary statistics
        summary_stats = {
            'total_attendee_instances': total_attendees,
            'unique_attendees': len(unique_attendees),
            'engaged_instances': engaged_attendees,
            'not_engaged_instances': not_engaged_attendees,
            'engagement_rate': (engaged_attendees / total_attendees * 100) if total_attendees > 0 else 0,
            'unique_meetings': attendee_df['Meeting_Subject'].nunique(),
            'unique_domains': attendee_df['Attendee_Domain'].nunique()
        }
        
        self.logger.info("Attendee analysis complete: %d total instances, %.1f%% engagement rate",
                        total_attendees, summary_stats['engagement_rate'])
        
        return {
            'attendee_matrix': attendee_df,
            'unique_attendees_summary': unique_attendees,
            'engagement_gaps': gaps,
            'summary_stats': summary_stats
        }
    
    @log_performance
    def generate_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive client mapping analysis"""
        self.logger.info("Generating comprehensive client mapping analysis")
        
        # Original domain-based analysis (KEEPING ALL EXISTING FUNCTIONALITY)
        df = self.data.copy()
        df['Client Domain'] = df.apply(self._get_client_domain, axis=1)
        df_clients = df[df['Client Domain'].isin(['unknown', 'internal']) == False]
        
        domain_analysis = {}
        if not df_clients.empty:
            # Create communication matrix
            client_stats = df_clients.groupby('Client Domain').agg({
                'Meeting Subject': 'nunique',
                'Email Subject': 'nunique',
                'Direction': 'count'
            }).rename(columns={'Meeting Subject': 'Meetings', 'Email Subject': 'Emails', 
                              'Direction': 'Total_Communications'})
            
            client_stats['Engagement_Score'] = (client_stats['Emails'] * 0.5 + 
                                               client_stats['Meetings'] * 2)
            client_stats = client_stats.sort_values('Engagement_Score', ascending=False)
            
            # Analyze gaps
            gaps = {
                'meetings_no_emails': client_stats[(client_stats['Meetings'] > 0) & 
                                                 (client_stats['Emails'] == 0)],
                'emails_no_meetings': client_stats[(client_stats['Meetings'] == 0) & 
                                                 (client_stats['Emails'] > 0)],
                'balanced': client_stats[(client_stats['Meetings'] > 0) & 
                                       (client_stats['Emails'] > 0)]
            }
            
            gap_summary = {
                'meetings_no_emails_count': len(gaps['meetings_no_emails']),
                'emails_no_meetings_count': len(gaps['emails_no_meetings']),
                'balanced_count': len(gaps['balanced']),
                'total_clients': len(client_stats)
            }
            
            # Calculate percentages
            total = gap_summary['total_clients']
            if total > 0:
                gap_summary.update({
                    'meetings_no_emails_pct': (gap_summary['meetings_no_emails_count'] / total) * 100,
                    'emails_no_meetings_pct': (gap_summary['emails_no_meetings_count'] / total) * 100,
                    'balanced_pct': (gap_summary['balanced_count'] / total) * 100
                })
            
            domain_analysis = {
                'communication_matrix': client_stats,
                'engagement_gaps': {'summary': gap_summary, **gaps},
                'total_domains': len(client_stats),
                'processed_data': df_clients
            }
        else:
            domain_analysis = {
                'communication_matrix': pd.DataFrame(),
                'engagement_gaps': {},
                'total_domains': 0,
                'processed_data': df_clients
            }
        
        # NEW: Attendee-level analysis
        attendee_analysis = self._generate_attendee_analysis()
        
        # Combine both analyses
        combined_analysis = {
            **domain_analysis,
            'attendee_analysis': attendee_analysis
        }
        
        self.logger.info("Combined analysis complete - domains: %d, attendee instances: %d",
                        domain_analysis['total_domains'],
                        attendee_analysis['summary_stats'].get('total_attendee_instances', 0))
        
        return combined_analysis
    
    def render_report(self) -> None:
        """Render enhanced client mapping report"""
        analysis = self.generate_analysis()
        
        self.render_header("Client Communication Mapping", 
                          "Email vs Meeting engagement gaps with attendee analysis", "🗺️")
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["📊 Domain Overview", "👥 Attendee Analysis", "🔍 Detailed Insights"])
        
        with tab1:
            # EXISTING FUNCTIONALITY - Domain-based analysis
            st.subheader("📊 Domain-Based Analysis")
            gaps = analysis['engagement_gaps'].get('summary', {})
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Client Domains", f"{analysis['total_domains']:,}")
            with col2:
                st.metric("Meetings Only", f"{gaps.get('meetings_no_emails_count', 0):,}")
            with col3:
                st.metric("Emails Only", f"{gaps.get('emails_no_meetings_count', 0):,}")
            with col4:
                st.metric("Balanced", f"{gaps.get('balanced_count', 0):,}")
            
            # Communication matrix
            matrix = analysis['communication_matrix']
            if not matrix.empty:
                st.subheader("📋 Top 20 Client Domains by Engagement")
                st.dataframe(matrix.head(20), use_container_width=True)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    top10 = matrix.head(10)
                    fig = px.bar(x=top10['Engagement_Score'], y=top10.index, 
                               orientation='h', title="Top 10 Domains by Engagement Score")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.scatter(matrix, x='Emails', y='Meetings', 
                                   title="Email vs Meeting Activity by Domain")
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # NEW FUNCTIONALITY - Attendee-level analysis
            attendee_data = analysis['attendee_analysis']
            stats = attendee_data['summary_stats']
            
            st.subheader("👥 Individual Attendee Engagement Analysis")
            
            # Attendee metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Attendee Instances", f"{stats.get('total_attendee_instances', 0):,}")
            with col2:
                st.metric("Unique Attendees", f"{stats.get('unique_attendees', 0):,}")
            with col3:
                st.metric("Engaged Instances", f"{stats.get('engaged_instances', 0):,}")
            with col4:
                engagement_rate = stats.get('engagement_rate', 0)
                st.metric("Engagement Rate", f"{engagement_rate:.1f}%")
            
            # Show engagement gaps matrix
            attendee_matrix = attendee_data['attendee_matrix']
            if not attendee_matrix.empty:
                st.subheader("🔍 Engagement Gaps - Attendees Without Email Contact")
                
                # Filter for non-engaged attendees
                not_engaged = attendee_matrix[attendee_matrix['Has_Email_Engagement'] == False]
                
                if not not_engaged.empty:
                    st.error(f"⚠️ Found {len(not_engaged)} attendee instances with NO email engagement:")
                    
                    # Display non-engaged attendees
                    display_cols = ['Meeting_Subject', 'Attendee_Email', 'In_Meeting', 'Engagement_Status']
                    st.dataframe(not_engaged[display_cols], use_container_width=True)
                    
                    # Group by domain to show which client companies have gaps
                    domain_gaps = not_engaged.groupby('Attendee_Domain').agg({
                        'Attendee_Email': 'nunique',
                        'Meeting_Subject': 'nunique'
                    }).reset_index()
                    domain_gaps.columns = ['Client Domain', 'Unique Attendees (No Email)', 'Meetings Involved']
                    domain_gaps = domain_gaps.sort_values('Unique Attendees (No Email)', ascending=False)
                    
                    st.subheader("🏢 Client Domains with Engagement Gaps")
                    st.dataframe(domain_gaps, use_container_width=True)
                    
                    # Visualization of gaps
                    fig = px.bar(domain_gaps.head(15), 
                               x='Client Domain', y='Unique Attendees (No Email)',
                               title="Top 15 Client Domains with Email Engagement Gaps")
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("✅ Excellent! All meeting attendees have email engagement.")
                
                # Engagement status pie chart
                status_counts = attendee_matrix['Engagement_Status'].value_counts()
                fig = px.pie(values=status_counts.values, names=status_counts.index,
                           title="Overall Attendee Engagement Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Combined insights
            st.subheader("💡 Key Insights & Compliance Gaps")
            
            # Domain-based insights (EXISTING)
            if analysis['total_domains'] > 0:
                gaps = analysis['engagement_gaps'].get('summary', {})
                meeting_gap_pct = gaps.get('meetings_no_emails_pct', 0)
                if meeting_gap_pct > 20:
                    st.warning(f"🔴 High domain gap: {meeting_gap_pct:.1f}% of client domains have meetings but no emails")
                elif meeting_gap_pct > 10:
                    st.info(f"🟡 Moderate domain gap: {meeting_gap_pct:.1f}% need email follow-up")
                
                balanced_pct = gaps.get('balanced_pct', 0)
                if balanced_pct > 70:
                    st.success(f"✅ Good domain balance: {balanced_pct:.1f}% have both meetings and emails")
            
            # Attendee-based insights (NEW)
            attendee_stats = analysis['attendee_analysis']['summary_stats']
            engagement_rate = attendee_stats.get('engagement_rate', 0)
            
            if engagement_rate < 50:
                st.error(f"🚨 Critical: Only {engagement_rate:.1f}% of attendee instances have email engagement!")
                st.write("**Compliance Risk:** High number of claimed meeting participants lack documented email communication.")
            elif engagement_rate < 75:
                st.warning(f"⚠️ Moderate risk: {engagement_rate:.1f}% attendee engagement rate")
                st.write("**Recommendation:** Increase email follow-up with meeting participants.")
            else:
                st.success(f"✅ Good engagement: {engagement_rate:.1f}% of attendees have email contact")
            
            # Summary recommendations
            st.subheader("📋 Compliance & Action Items")
            
            not_engaged_count = attendee_stats.get('total_attendee_instances', 0) - attendee_stats.get('engaged_instances', 0)
            if not_engaged_count > 0:
                st.write(f"""
                **Audit Findings:**
                - {not_engaged_count} attendee instances lack email documentation
                - This represents a {100 - engagement_rate:.1f}% gap between claimed engagement and actual contact
                - Review these cases for compliance and engagement validation
                
                **Recommended Actions:**
                1. Follow up with non-engaged attendees via email
                2. Document reasons for lack of email communication
                3. Implement email follow-up protocols for all meeting attendees
                """)
            else:
                st.success("✅ All meeting attendees have documented email engagement - excellent compliance!")
    
    def get_report_summary(self) -> Dict[str, Any]:
        """Get enhanced summary for dashboard"""
        analysis = self.generate_analysis()
        gaps = analysis['engagement_gaps'].get('summary', {})
        attendee_stats = analysis['attendee_analysis']['summary_stats']
        
        return {
            # Existing metrics
            'total_client_domains': analysis['total_domains'],
            'communication_gaps': gaps.get('meetings_no_emails_count', 0),
            'balanced_engagement': gaps.get('balanced_count', 0),
            'engagement_coverage': gaps.get('balanced_pct', 0),
            
            # New attendee metrics
            'total_attendee_instances': attendee_stats.get('total_attendee_instances', 0),
            'attendee_engagement_rate': attendee_stats.get('engagement_rate', 0),
            'unique_attendees': attendee_stats.get('unique_attendees', 0),
            'compliance_risk_level': 'High' if attendee_stats.get('engagement_rate', 0) < 50 else 
                                   'Medium' if attendee_stats.get('engagement_rate', 0) < 75 else 'Low'
        }