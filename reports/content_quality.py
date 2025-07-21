"""
Enhanced Engagement Quality Summary Report - Keyword-Focused Analysis
Reviews email content with advanced keyword analytics and visualization
"""

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from typing import Dict, Any, List, Tuple
import re
from collections import Counter, defaultdict

from .base import BaseReport
from logger import log_performance


class ContentQualityReport(BaseReport):
    """
    Enhanced Report 4: Engagement Quality Summary - Keyword-Focused
    
    Purpose: Deep analysis of email content through advanced keyword analytics,
    co-occurrence patterns, and semantic clustering for business intelligence.
    Output: Comprehensive keyword insights with interactive visualizations
    Value: Identifies communication patterns, business focus areas, and content gaps
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.info("Initializing Enhanced Content Quality Report with keyword focus")
        
    def get_required_columns(self) -> List[str]:
        """Get required columns for content quality analysis"""
        return [
            'Email Subject',
            'Email Body Content',
            'Meeting Subject',
            'Direction',
            'Email Time (BST/GMT)'
        ]
    
    @log_performance
    def generate_analysis(self) -> Dict[str, Any]:
        """
        Generate comprehensive keyword-focused content analysis
        
        Returns:
            Dict[str, Any]: Analysis results with enhanced keyword insights
        """
        self.logger.info("Generating enhanced keyword-focused content analysis")
        
        # Enhanced content analysis with detailed keyword tracking
        content_analysis = self._calculate_enhanced_substance_scores()
        
        # Deep keyword analysis
        keyword_analysis = self._generate_comprehensive_keyword_analysis(content_analysis)
        
        # Keyword co-occurrence and relationship analysis
        cooccurrence_analysis = self._analyze_keyword_cooccurrence(content_analysis)
        
        # Temporal keyword trends
        temporal_analysis = self._analyze_keyword_temporal_trends(content_analysis)
        
        # Keyword-based email categorization
        email_categorization = self._categorize_emails_by_keywords(content_analysis)
        
        # Business context analysis
        business_context = self._analyze_business_context(content_analysis)
        
        # Original quality analysis (maintaining existing functionality)
        quality_categories = self._categorize_content_quality(content_analysis)
        direction_analysis = self._analyze_by_direction(content_analysis)
        
        results = {
            'content_data': content_analysis,
            'keyword_analysis': keyword_analysis,
            'cooccurrence_analysis': cooccurrence_analysis,
            'temporal_analysis': temporal_analysis,
            'email_categorization': email_categorization,
            'business_context': business_context,
            'quality_categories': quality_categories,
            'direction_analysis': direction_analysis,
            'total_emails': len(content_analysis),
            'avg_substance_score': content_analysis['Substance_Score'].mean() if len(content_analysis) > 0 else 0
        }
        
        self.logger.info("Enhanced keyword analysis completed: %d emails, %d unique keywords",
                        results['total_emails'], 
                        results['keyword_analysis']['unique_keywords_count'])
        
        return results
    
    def _calculate_enhanced_substance_scores(self) -> pd.DataFrame:
        """
        Calculate enhanced substance scores with detailed keyword tracking
        
        Returns:
            pd.DataFrame: DataFrame with enhanced keyword analysis
        """
        self.logger.info("Calculating enhanced substance scores with detailed keyword tracking")
        
        df_analysis = self.data.copy()
        
        # Enhanced keyword extraction with positions and context
        df_analysis['Subject_Keywords'] = df_analysis['Email Subject'].apply(
            lambda x: self._extract_keywords_with_context(x, 'subject')
        )
        
        df_analysis['Body_Keywords'] = df_analysis['Email Body Content'].apply(
            lambda x: self._extract_keywords_with_context(x, 'body')
        )
        
        # Calculate enhanced scores
        df_analysis['Subject_Score'] = df_analysis['Subject_Keywords'].apply(
            lambda x: self._calculate_keyword_score_from_context(x)
        )
        
        df_analysis['Body_Score'] = df_analysis['Body_Keywords'].apply(
            lambda x: self._calculate_keyword_score_from_context(x)
        )
        
        # Enhanced combined substance score
        df_analysis['Substance_Score'] = (
            df_analysis['Subject_Score'] * 0.3 + 
            df_analysis['Body_Score'] * 0.7
        )
        
        # Comprehensive keyword analysis
        df_analysis['All_Keywords'] = df_analysis.apply(
            lambda row: self._merge_keyword_contexts(row['Subject_Keywords'], row['Body_Keywords']),
            axis=1
        )
        
        # Keyword categories found
        df_analysis['Keyword_Categories'] = df_analysis['All_Keywords'].apply(
            lambda x: self._identify_keyword_categories(x)
        )
        
        # Business intensity metrics
        df_analysis['Business_Intensity'] = df_analysis['All_Keywords'].apply(
            lambda x: self._calculate_business_intensity(x)
        )
        
        # Content type classification
        df_analysis['Content_Type'] = df_analysis['All_Keywords'].apply(
            lambda x: self._classify_content_type(x)
        )
        
        # Enhanced content length analysis
        df_analysis['Subject_Length'] = df_analysis['Email Subject'].astype(str).str.len()
        df_analysis['Body_Length'] = df_analysis['Email Body Content'].astype(str).str.len()
        df_analysis['Keyword_Density'] = df_analysis.apply(
            lambda row: len(row['All_Keywords']) / max(row['Body_Length'], 1) * 100,
            axis=1
        )
        
        self.logger.info("Enhanced substance scores calculated for %d emails", len(df_analysis))
        return df_analysis
    
    def _extract_keywords_with_context(self, text: str, content_type: str = 'general') -> List[Dict[str, Any]]:
        """
        Extract keywords with context information
        
        Args:
            text: Text to analyze
            content_type: Type of content ('subject', 'body', 'general')
            
        Returns:
            List[Dict]: List of keyword dictionaries with context
        """
        if pd.isna(text) or text == 'nan':
            return []
        
        text_lower = str(text).lower()
        keywords_with_context = []
        
        for category, keywords in self.config.ENGAGEMENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Find all occurrences
                    start = 0
                    while True:
                        pos = text_lower.find(keyword, start)
                        if pos == -1:
                            break
                        
                        # Extract context (surrounding words)
                        context_start = max(0, pos - 30)
                        context_end = min(len(text), pos + len(keyword) + 30)
                        context = text[context_start:context_end].strip()
                        
                        keywords_with_context.append({
                            'keyword': keyword,
                            'category': category,
                            'position': pos,
                            'context': context,
                            'content_type': content_type,
                            'length': len(keyword)
                        })
                        
                        start = pos + 1
        
        return keywords_with_context
    
    def _calculate_keyword_score_from_context(self, keyword_contexts: List[Dict[str, Any]]) -> int:
        """
        Calculate score from keyword contexts with enhanced weighting
        
        Args:
            keyword_contexts: List of keyword context dictionaries
            
        Returns:
            int: Enhanced quality score
        """
        if not keyword_contexts:
            return 0
        
        total_score = 0
        weights = self.config.get_keyword_score_weights()
        categories_found = set()
        
        for kw_ctx in keyword_contexts:
            category = kw_ctx['category']
            weight = weights.get(category, 1)
            
            # Enhanced scoring based on context
            score = weight
            
            # Bonus for subject keywords
            if kw_ctx['content_type'] == 'subject':
                score *= 1.3
            
            # Bonus for longer keywords (more specific)
            if kw_ctx['length'] > 10:
                score *= 1.2
            
            total_score += score
            categories_found.add(category)
        
        # Diversity bonus
        if len(categories_found) >= 2:
            total_score += 3
        if len(categories_found) >= 3:
            total_score += 2
        
        return int(total_score)
    
    def _merge_keyword_contexts(self, subject_keywords: List[Dict], body_keywords: List[Dict]) -> List[Dict]:
        """Merge keyword contexts from subject and body"""
        all_keywords = subject_keywords.copy()
        all_keywords.extend(body_keywords)
        
        # Remove duplicates while preserving context
        seen = set()
        unique_keywords = []
        
        for kw in all_keywords:
            key = (kw['keyword'], kw['category'])
            if key not in seen:
                unique_keywords.append(kw)
                seen.add(key)
        
        return unique_keywords
    
    def _identify_keyword_categories(self, keyword_contexts: List[Dict]) -> List[str]:
        """Identify categories present in keyword contexts"""
        categories = set()
        for kw_ctx in keyword_contexts:
            categories.add(kw_ctx['category'])
        return list(categories)
    
    def _calculate_business_intensity(self, keyword_contexts: List[Dict]) -> float:
        """Calculate business intensity score based on keyword types"""
        if not keyword_contexts:
            return 0.0
        
        # Weight different categories for business intensity
        intensity_weights = {
            'action_items': 3.0,
            'proposals': 2.8,
            'commitments': 2.5,
            'follow_up': 2.0,
            'meetings': 1.8,
            'confirmations': 1.5,
            'generic': 1.0
        }
        
        total_intensity = 0
        for kw_ctx in keyword_contexts:
            category = kw_ctx['category']
            weight = intensity_weights.get(category, 1.0)
            total_intensity += weight
        
        return round(total_intensity / len(keyword_contexts), 2)
    
    def _classify_content_type(self, keyword_contexts: List[Dict]) -> str:
        """Classify email content type based on dominant keywords"""
        if not keyword_contexts:
            return 'Generic'
        
        category_counts = defaultdict(int)
        for kw_ctx in keyword_contexts:
            category_counts[kw_ctx['category']] += 1
        
        if not category_counts:
            return 'Generic'
        
        dominant_category = max(category_counts, key=category_counts.get)
        
        classification_map = {
            'action_items': 'Action-Oriented',
            'proposals': 'Proposal/Sales',
            'commitments': 'Commitment-Based',
            'follow_up': 'Follow-Up',
            'meetings': 'Meeting-Related',
            'confirmations': 'Confirmatory',
            'generic': 'Generic'
        }
        
        return classification_map.get(dominant_category, 'Generic')
    
    def _generate_comprehensive_keyword_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive keyword analysis with enhanced metrics
        
        Args:
            df: DataFrame with keyword analysis
            
        Returns:
            Dict[str, Any]: Comprehensive keyword insights
        """
        self.logger.info("Generating comprehensive keyword analysis")
        
        # Collect all keywords with metadata
        all_keywords_detailed = []
        for _, row in df.iterrows():
            for kw_ctx in row['All_Keywords']:
                all_keywords_detailed.append({
                    **kw_ctx,
                    'email_index': row.name,
                    'direction': row.get('Direction', 'Unknown'),
                    'substance_score': row['Substance_Score']
                })
        
        keywords_df = pd.DataFrame(all_keywords_detailed)
        
        if keywords_df.empty:
            return self._empty_keyword_analysis()
        
        # Enhanced keyword metrics
        analysis = {
            'detailed_keywords': keywords_df,
            'unique_keywords_count': keywords_df['keyword'].nunique(),
            'total_keyword_instances': len(keywords_df),
            'avg_keywords_per_email': len(keywords_df) / len(df) if len(df) > 0 else 0
        }
        
        # Keyword frequency analysis
        analysis['frequency_analysis'] = self._analyze_keyword_frequency(keywords_df)
        
        # Category-based analysis
        analysis['category_analysis'] = self._analyze_keyword_categories(keywords_df)
        
        # Content type distribution
        analysis['content_type_distribution'] = df['Content_Type'].value_counts().to_dict()
        
        # Business intensity analysis
        analysis['business_intensity'] = {
            'avg_intensity': df['Business_Intensity'].mean(),
            'intensity_distribution': df['Business_Intensity'].describe().to_dict(),
            'high_intensity_emails': len(df[df['Business_Intensity'] > 2.0])
        }
        
        # Keyword density analysis
        analysis['density_analysis'] = {
            'avg_density': df['Keyword_Density'].mean(),
            'density_quartiles': df['Keyword_Density'].quantile([0.25, 0.5, 0.75]).to_dict()
        }
        
        # Direction-based keyword analysis
        analysis['direction_keywords'] = self._analyze_keywords_by_direction(keywords_df)
        
        self.logger.info("Comprehensive keyword analysis completed")
        return analysis
    
    def _analyze_keyword_frequency(self, keywords_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze keyword frequency patterns"""
        if keywords_df.empty:
            return {}
        
        # Top keywords overall
        keyword_counts = keywords_df['keyword'].value_counts()
        
        # Top keywords by category
        category_keywords = {}
        for category in keywords_df['category'].unique():
            cat_keywords = keywords_df[keywords_df['category'] == category]['keyword'].value_counts()
            category_keywords[category] = cat_keywords.head(10).to_dict()
        
        # Keyword rarity analysis
        rare_keywords = keyword_counts[keyword_counts == 1].index.tolist()
        common_keywords = keyword_counts[keyword_counts >= 5].index.tolist()
        
        return {
            'top_keywords_overall': keyword_counts.head(20).to_dict(),
            'top_keywords_by_category': category_keywords,
            'rare_keywords': rare_keywords[:20],  # Limit for display
            'common_keywords': common_keywords,
            'keyword_diversity': len(keyword_counts) / keyword_counts.sum() if keyword_counts.sum() > 0 else 0
        }
    
    def _analyze_keyword_categories(self, keywords_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze keyword distribution by categories"""
        if keywords_df.empty:
            return {}
        
        category_stats = keywords_df.groupby('category').agg({
            'keyword': ['count', 'nunique'],
            'substance_score': 'mean'
        }).round(2)
        
        category_stats.columns = ['Total_Instances', 'Unique_Keywords', 'Avg_Email_Score']
        category_stats = category_stats.reset_index()
        
        # Category distribution percentages
        category_distribution = keywords_df['category'].value_counts(normalize=True) * 100
        
        return {
            'category_statistics': category_stats.to_dict('records'),
            'category_distribution_pct': category_distribution.to_dict(),
            'dominant_category': category_distribution.index[0] if len(category_distribution) > 0 else None
        }
    
    def _analyze_keywords_by_direction(self, keywords_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze keyword patterns by email direction"""
        if keywords_df.empty or 'direction' not in keywords_df.columns:
            return {}
        
        direction_analysis = {}
        
        for direction in keywords_df['direction'].unique():
            if pd.notna(direction):
                dir_keywords = keywords_df[keywords_df['direction'] == direction]
                
                direction_analysis[direction] = {
                    'total_keywords': len(dir_keywords),
                    'unique_keywords': dir_keywords['keyword'].nunique(),
                    'top_keywords': dir_keywords['keyword'].value_counts().head(10).to_dict(),
                    'avg_substance_score': dir_keywords['substance_score'].mean(),
                    'category_distribution': dir_keywords['category'].value_counts().to_dict()
                }
        
        return direction_analysis
    
    def _analyze_keyword_cooccurrence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze keyword co-occurrence patterns
        
        Args:
            df: DataFrame with keyword analysis
            
        Returns:
            Dict[str, Any]: Co-occurrence analysis
        """
        self.logger.info("Analyzing keyword co-occurrence patterns")
        
        # Build co-occurrence matrix
        email_keywords = {}
        for idx, row in df.iterrows():
            keywords = [kw['keyword'] for kw in row['All_Keywords']]
            if keywords:
                email_keywords[idx] = set(keywords)
        
        if not email_keywords:
            return {'cooccurrence_matrix': pd.DataFrame(), 'keyword_pairs': []}
        
        # Calculate co-occurrence for frequently occurring keywords
        all_keywords = [kw for kw_set in email_keywords.values() for kw in kw_set]
        keyword_freq = Counter(all_keywords)
        frequent_keywords = [kw for kw, count in keyword_freq.items() if count >= 3]
        
        if len(frequent_keywords) < 2:
            return {'cooccurrence_matrix': pd.DataFrame(), 'keyword_pairs': []}
        
        # Build co-occurrence matrix
        cooccurrence_matrix = pd.DataFrame(0, index=frequent_keywords, columns=frequent_keywords)
        
        for email_kws in email_keywords.values():
            email_frequent = [kw for kw in email_kws if kw in frequent_keywords]
            for i, kw1 in enumerate(email_frequent):
                for kw2 in email_frequent[i+1:]:
                    cooccurrence_matrix.loc[kw1, kw2] += 1
                    cooccurrence_matrix.loc[kw2, kw1] += 1
        
        # Find strongest keyword pairs
        keyword_pairs = []
        for i in range(len(frequent_keywords)):
            for j in range(i+1, len(frequent_keywords)):
                kw1, kw2 = frequent_keywords[i], frequent_keywords[j]
                count = cooccurrence_matrix.loc[kw1, kw2]
                if count > 0:
                    keyword_pairs.append({
                        'keyword1': kw1,
                        'keyword2': kw2,
                        'cooccurrence_count': count,
                        'strength': count / min(keyword_freq[kw1], keyword_freq[kw2])
                    })
        
        keyword_pairs.sort(key=lambda x: x['strength'], reverse=True)
        
        return {
            'cooccurrence_matrix': cooccurrence_matrix,
            'keyword_pairs': keyword_pairs[:20],  # Top 20 pairs
            'network_data': self._prepare_network_data(keyword_pairs[:15])
        }
    
    def _prepare_network_data(self, keyword_pairs: List[Dict]) -> Dict[str, Any]:
        """Prepare data for network visualization"""
        if not keyword_pairs:
            return {'nodes': [], 'edges': []}
        
        # Create nodes and edges for network graph
        nodes = set()
        edges = []
        
        for pair in keyword_pairs:
            nodes.add(pair['keyword1'])
            nodes.add(pair['keyword2'])
            edges.append({
                'source': pair['keyword1'],
                'target': pair['keyword2'],
                'weight': pair['cooccurrence_count'],
                'strength': pair['strength']
            })
        
        nodes_list = [{'id': node, 'label': node} for node in nodes]
        
        return {'nodes': nodes_list, 'edges': edges}
    
    def _analyze_keyword_temporal_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze keyword trends over time
        
        Args:
            df: DataFrame with time and keyword data
            
        Returns:
            Dict[str, Any]: Temporal analysis
        """
        if 'Email Time (BST/GMT)' not in df.columns:
            return {}
        
        # Prepare temporal data
        df_temporal = df.copy()
        df_temporal['Email_Date'] = pd.to_datetime(df_temporal['Email Time (BST/GMT)'], errors='coerce')
        df_temporal = df_temporal.dropna(subset=['Email_Date'])
        
        if len(df_temporal) == 0:
            return {}
        
        # Extract keywords by time period
        df_temporal['Month'] = df_temporal['Email_Date'].dt.to_period('M')
        
        # Keyword trends by month
        monthly_keyword_trends = {}
        for month in df_temporal['Month'].unique():
            month_data = df_temporal[df_temporal['Month'] == month]
            month_keywords = []
            
            for _, row in month_data.iterrows():
                month_keywords.extend([kw['keyword'] for kw in row['All_Keywords']])
            
            keyword_counts = Counter(month_keywords)
            monthly_keyword_trends[str(month)] = dict(keyword_counts.most_common(10))
        
        # Category trends
        monthly_category_trends = df_temporal.groupby('Month')['Keyword_Categories'].apply(
            lambda x: [cat for cats in x for cat in cats]
        ).apply(lambda x: dict(Counter(x))).to_dict()
        
        # Convert period keys to strings
        monthly_category_trends = {str(k): v for k, v in monthly_category_trends.items()}
        
        return {
            'monthly_keyword_trends': monthly_keyword_trends,
            'monthly_category_trends': monthly_category_trends,
            'trend_summary': self._summarize_keyword_trends(monthly_keyword_trends)
        }
    
    def _summarize_keyword_trends(self, monthly_trends: Dict[str, Dict]) -> Dict[str, Any]:
        """Summarize keyword trends"""
        if not monthly_trends:
            return {}
        
        # Find consistently appearing keywords
        all_months = list(monthly_trends.keys())
        consistent_keywords = set(monthly_trends[all_months[0]].keys())
        
        for month_trends in monthly_trends.values():
            consistent_keywords &= set(month_trends.keys())
        
        # Find trending keywords (appearing in later months)
        if len(all_months) >= 2:
            recent_months = all_months[-2:]
            trending = set()
            for month in recent_months:
                trending.update(monthly_trends[month].keys())
            
            early_months = all_months[:-2] if len(all_months) > 2 else []
            early_keywords = set()
            for month in early_months:
                early_keywords.update(monthly_trends[month].keys())
            
            emerging_keywords = trending - early_keywords
        else:
            emerging_keywords = set()
        
        return {
            'consistent_keywords': list(consistent_keywords)[:10],
            'emerging_keywords': list(emerging_keywords)[:10],
            'total_periods': len(all_months)
        }
    
    def _categorize_emails_by_keywords(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Categorize emails based on keyword patterns"""
        if df.empty:
            return {}
        
        # Enhanced categorization based on keyword categories and intensity
        categories = {
            'High Business Value': df[
                (df['Business_Intensity'] >= 2.5) & 
                (df['Substance_Score'] >= 8)
            ],
            'Strategic Communications': df[
                df['Keyword_Categories'].apply(
                    lambda cats: any(cat in ['proposals', 'commitments'] for cat in cats)
                )
            ],
            'Operational': df[
                df['Keyword_Categories'].apply(
                    lambda cats: any(cat in ['action_items', 'follow_up'] for cat in cats)
                ) & (df['Business_Intensity'] < 2.5)
            ],
            'Routine/Administrative': df[
                (df['Business_Intensity'] < 1.5) & 
                (df['Substance_Score'] < 4)
            ]
        }
        
        # Remove overlaps (prioritize higher value categories)
        processed_indices = set()
        final_categories = {}
        
        for cat_name in ['High Business Value', 'Strategic Communications', 'Operational', 'Routine/Administrative']:
            cat_df = categories[cat_name]
            # Remove already processed emails
            cat_df = cat_df[~cat_df.index.isin(processed_indices)]
            final_categories[cat_name] = cat_df
            processed_indices.update(cat_df.index)
        
        # Calculate statistics
        total_emails = len(df)
        category_stats = {}
        
        for cat_name, cat_df in final_categories.items():
            count = len(cat_df)
            percentage = (count / total_emails) * 100 if total_emails > 0 else 0
            avg_score = cat_df['Substance_Score'].mean() if len(cat_df) > 0 else 0
            avg_intensity = cat_df['Business_Intensity'].mean() if len(cat_df) > 0 else 0
            
            category_stats[cat_name] = {
                'count': count,
                'percentage': percentage,
                'avg_substance_score': avg_score,
                'avg_business_intensity': avg_intensity
            }
        
        return {
            'categories': final_categories,
            'statistics': category_stats
        }
    
    def _analyze_business_context(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze business context from keyword patterns"""
        if df.empty:
            return {}
        
        # Identify business themes
        theme_keywords = {
            'Sales & Proposals': ['proposal', 'quote', 'pricing', 'contract', 'deal', 'sales'],
            'Project Management': ['deadline', 'milestone', 'deliverable', 'timeline', 'project'],
            'Client Relations': ['client', 'customer', 'relationship', 'satisfaction', 'feedback'],
            'Operations': ['process', 'workflow', 'efficiency', 'operations', 'implementation'],
            'Strategic Planning': ['strategy', 'planning', 'goals', 'objectives', 'vision']
        }
        
        theme_analysis = {}
        for theme, keywords in theme_keywords.items():
            theme_emails = df[
                df['All_Keywords'].apply(
                    lambda kw_list: any(
                        any(tk in kw['keyword'] for tk in keywords) 
                        for kw in kw_list
                    )
                )
            ]
            
            if len(theme_emails) > 0:
                theme_analysis[theme] = {
                    'email_count': len(theme_emails),
                    'avg_substance_score': theme_emails['Substance_Score'].mean(),
                    'percentage': (len(theme_emails) / len(df)) * 100
                }
        
        return {
            'business_themes': theme_analysis,
            'dominant_theme': max(theme_analysis.keys(), key=lambda k: theme_analysis[k]['email_count']) if theme_analysis else None
        }
    
    def _empty_keyword_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure"""
        return {
            'detailed_keywords': pd.DataFrame(),
            'unique_keywords_count': 0,
            'total_keyword_instances': 0,
            'avg_keywords_per_email': 0,
            'frequency_analysis': {},
            'category_analysis': {},
            'content_type_distribution': {},
            'business_intensity': {'avg_intensity': 0},
            'density_analysis': {'avg_density': 0},
            'direction_keywords': {}
        }
    
    # Keep existing methods for backward compatibility
    def _categorize_content_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Categorize emails by content quality levels (existing functionality)"""
        # Define quality thresholds
        high_threshold = 8
        medium_threshold = 4
        low_threshold = 1
        
        # Categorize emails
        categories = {
            'high_quality': df[df['Substance_Score'] >= high_threshold],
            'medium_quality': df[
                (df['Substance_Score'] >= medium_threshold) & 
                (df['Substance_Score'] < high_threshold)
            ],
            'low_quality': df[
                (df['Substance_Score'] >= low_threshold) & 
                (df['Substance_Score'] < medium_threshold)
            ],
            'minimal_quality': df[df['Substance_Score'] < low_threshold]
        }
        
        # Calculate statistics
        total_emails = len(df)
        category_stats = {}
        
        for category_name, category_df in categories.items():
            count = len(category_df)
            percentage = (count / total_emails) * 100 if total_emails > 0 else 0
            avg_score = category_df['Substance_Score'].mean() if len(category_df) > 0 else 0
            
            category_stats[category_name] = {
                'count': count,
                'percentage': percentage,
                'avg_score': avg_score,
                'data': category_df
            }
        
        return {
            'categories': categories,
            'statistics': category_stats,
            'thresholds': {
                'high': high_threshold,
                'medium': medium_threshold,
                'low': low_threshold
            }
        }
    
    def _analyze_by_direction(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze content quality by email direction (existing functionality)"""
        if 'Direction' not in df.columns:
            return {}
        
        direction_stats = {}
        
        for direction in df['Direction'].unique():
            if pd.notna(direction):
                direction_data = df[df['Direction'] == direction]
                
                if len(direction_data) > 0:
                    direction_stats[direction] = {
                        'count': len(direction_data),
                        'avg_substance_score': direction_data['Substance_Score'].mean(),
                        'avg_business_intensity': direction_data['Business_Intensity'].mean(),
                        'unique_keywords': sum(len(kw_list) for kw_list in direction_data['All_Keywords']),
                        'high_quality_pct': (direction_data['Substance_Score'] >= 8).mean() * 100,
                        'low_quality_pct': (direction_data['Substance_Score'] < 2).mean() * 100
                    }
        
        return direction_stats
    
    def render_report(self) -> None:
        """Render the enhanced keyword-focused content quality report"""
        self.logger.info("Rendering enhanced keyword-focused content quality report")
        
        # Generate analysis
        analysis = self.generate_analysis()
        
        # Render header
        self.render_header(
            "Enhanced Content Quality Analysis",
            "Deep keyword analytics and business intelligence from email communications",
            "🔍"
        )
        
        # Create main tabs with keyword focus
        main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
            "🔑 Keyword Analytics", 
            "📊 Business Intelligence", 
            "🕸️ Relationship Analysis", 
            "📈 Traditional Metrics"
        ])
        
        with main_tab1:
            self._render_keyword_analytics_tab(analysis)
        
        with main_tab2:
            self._render_business_intelligence_tab(analysis)
        
        with main_tab3:
            self._render_relationship_analysis_tab(analysis)
        
        with main_tab4:
            self._render_traditional_metrics_tab(analysis)
        
        self.logger.info("Enhanced keyword-focused report rendered successfully")
    
    def _render_keyword_analytics_tab(self, analysis: Dict[str, Any]) -> None:
        """Render comprehensive keyword analytics"""
        st.header("🔑 Advanced Keyword Analytics")
        
        keyword_analysis = analysis['keyword_analysis']
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Unique Keywords", 
                f"{keyword_analysis['unique_keywords_count']:,}",
                help="Total number of unique business keywords identified"
            )
        
        with col2:
            st.metric(
                "Keyword Instances", 
                f"{keyword_analysis['total_keyword_instances']:,}",
                help="Total occurrences of all keywords across emails"
            )
        
        with col3:
            avg_per_email = keyword_analysis['avg_keywords_per_email']
            st.metric(
                "Avg Keywords/Email", 
                f"{avg_per_email:.1f}",
                help="Average number of business keywords per email"
            )
        
        with col4:
            density = keyword_analysis.get('density_analysis', {}).get('avg_density', 0)
            st.metric(
                "Keyword Density", 
                f"{density:.2f}%",
                help="Average percentage of text that consists of business keywords"
            )
        
        # Keyword frequency analysis
        st.subheader("📊 Keyword Frequency Analysis")
        
        freq_analysis = keyword_analysis.get('frequency_analysis', {})
        
        if freq_analysis.get('top_keywords_overall'):
            col1, col2 = st.columns(2)
            
            with col1:
                # Top keywords bar chart
                top_keywords = freq_analysis['top_keywords_overall']
                keywords_df = pd.DataFrame(
                    list(top_keywords.items())[:15],
                    columns=['Keyword', 'Frequency']
                )
                
                fig = px.bar(
                    keywords_df,
                    x='Frequency',
                    y='Keyword',
                    orientation='h',
                    title="Top 15 Business Keywords",
                    color='Frequency',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Category distribution
                cat_analysis = keyword_analysis.get('category_analysis', {})
                cat_dist = cat_analysis.get('category_distribution_pct', {})
                
                if cat_dist:
                    fig = px.pie(
                        values=list(cat_dist.values()),
                        names=[cat.replace('_', ' ').title() for cat in cat_dist.keys()],
                        title="Keyword Category Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Keywords by category detailed view
        st.subheader("🏷️ Keywords by Category")
        
        if freq_analysis.get('top_keywords_by_category'):
            for category, keywords in freq_analysis['top_keywords_by_category'].items():
                if keywords:  # Only show categories with keywords
                    with st.expander(f"📂 {category.replace('_', ' ').title()} ({len(keywords)} keywords)"):
                        # Create columns for better display
                        keyword_list = list(keywords.items())[:10]  # Top 10 per category
                        
                        if len(keyword_list) > 5:
                            col1, col2 = st.columns(2)
                            mid = len(keyword_list) // 2
                            
                            with col1:
                                for kw, count in keyword_list[:mid]:
                                    st.write(f"• **{kw}**: {count} times")
                            
                            with col2:
                                for kw, count in keyword_list[mid:]:
                                    st.write(f"• **{kw}**: {count} times")
                        else:
                            for kw, count in keyword_list:
                                st.write(f"• **{kw}**: {count} times")
        
        # Business intensity analysis
        st.subheader("💼 Business Intensity Analysis")
        
        business_intensity = keyword_analysis.get('business_intensity', {})
        avg_intensity = business_intensity.get('avg_intensity', 0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Average Business Intensity", f"{avg_intensity:.2f}")
            
            if avg_intensity >= 2.5:
                st.success("🟢 High business intensity - substantive communications")
            elif avg_intensity >= 1.5:
                st.warning("🟡 Moderate business intensity - mixed communications")
            else:
                st.error("🔴 Low business intensity - mostly routine communications")
        
        with col2:
            high_intensity = business_intensity.get('high_intensity_emails', 0)
            total_emails = analysis['total_emails']
            intensity_pct = (high_intensity / total_emails * 100) if total_emails > 0 else 0
            
            st.metric("High Intensity Emails", f"{high_intensity} ({intensity_pct:.1f}%)")
    
    def _render_business_intelligence_tab(self, analysis: Dict[str, Any]) -> None:
        """Render business intelligence insights"""
        st.header("📊 Business Intelligence & Content Categorization")
        
        # Email categorization by business value
        email_categorization = analysis.get('email_categorization', {})
        
        if email_categorization:
            st.subheader("💼 Email Categorization by Business Value")
            
            # Display category statistics
            category_stats = email_categorization.get('statistics', {})
            
            if category_stats:
                # Create visualization of categories
                cat_data = []
                for cat_name, stats in category_stats.items():
                    cat_data.append({
                        'Category': cat_name,
                        'Count': stats['count'],
                        'Percentage': stats['percentage'],
                        'Avg Substance Score': stats['avg_substance_score'],
                        'Avg Business Intensity': stats.get('avg_business_intensity', 0)
                    })
                
                cat_df = pd.DataFrame(cat_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Category distribution pie chart
                    fig = px.pie(
                        cat_df,
                        names='Category',
                        values='Count',
                        title="Email Distribution by Business Value",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Category quality comparison
                    fig = px.bar(
                        cat_df,
                        x='Category',
                        y='Avg Substance Score',
                        title="Average Substance Score by Category",
                        color='Avg Business Intensity',
                        color_continuous_scale='viridis'
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed category table
                st.subheader("📋 Detailed Category Breakdown")
                st.dataframe(cat_df, use_container_width=True, hide_index=True)
        
        # Business context analysis
        business_context = analysis.get('business_context', {})
        
        if business_context.get('business_themes'):
            st.subheader("🏢 Business Theme Analysis")
            
            themes = business_context['business_themes']
            theme_data = []
            
            for theme, stats in themes.items():
                theme_data.append({
                    'Business Theme': theme,
                    'Email Count': stats['email_count'],
                    'Percentage': stats['percentage'],
                    'Avg Substance Score': stats['avg_substance_score']
                })
            
            theme_df = pd.DataFrame(theme_data)
            theme_df = theme_df.sort_values('Email Count', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    theme_df,
                    x='Business Theme',
                    y='Email Count',
                    title="Email Volume by Business Theme",
                    color='Avg Substance Score',
                    color_continuous_scale='viridis'
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.dataframe(theme_df, use_container_width=True, hide_index=True)
            
            # Highlight dominant theme
            dominant_theme = business_context.get('dominant_theme')
            if dominant_theme:
                st.success(f"🎯 **Dominant Business Theme**: {dominant_theme}")
        
        # Content type distribution
        content_types = analysis['keyword_analysis'].get('content_type_distribution', {})
        
        if content_types:
            st.subheader("📝 Content Type Distribution")
            
            content_df = pd.DataFrame(
                list(content_types.items()),
                columns=['Content Type', 'Count']
            )
            
            fig = px.bar(
                content_df,
                x='Content Type',
                y='Count',
                title="Distribution of Email Content Types",
                color='Count',
                color_continuous_scale='plasma'
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_relationship_analysis_tab(self, analysis: Dict[str, Any]) -> None:
        """Render keyword relationship and co-occurrence analysis"""
        st.header("🕸️ Keyword Relationships & Co-occurrence Analysis")
        
        cooccurrence_analysis = analysis.get('cooccurrence_analysis', {})
        
        if cooccurrence_analysis.get('keyword_pairs'):
            st.subheader("🔗 Keyword Co-occurrence Patterns")
            
            keyword_pairs = cooccurrence_analysis['keyword_pairs']
            
            # Top keyword pairs
            pairs_data = []
            for pair in keyword_pairs[:15]:
                pairs_data.append({
                    'Keyword 1': pair['keyword1'],
                    'Keyword 2': pair['keyword2'],
                    'Co-occurrence Count': pair['cooccurrence_count'],
                    'Relationship Strength': f"{pair['strength']:.3f}"
                })
            
            pairs_df = pd.DataFrame(pairs_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top Keyword Pairs by Co-occurrence:**")
                st.dataframe(pairs_df, use_container_width=True, hide_index=True)
            
            with col2:
                # Visualize top pairs
                if len(pairs_df) > 0:
                    pairs_df['Pair'] = pairs_df['Keyword 1'] + ' + ' + pairs_df['Keyword 2']
                    fig = px.bar(
                        pairs_df.head(10),
                        x='Co-occurrence Count',
                        y='Pair',
                        orientation='h',
                        title="Top 10 Keyword Pairs",
                        color='Co-occurrence Count',
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Network visualization concept
            st.subheader("🌐 Keyword Relationship Network")
            st.info("""
            **Network Analysis Insights:**
            
            Strong keyword co-occurrences indicate:
            - Common business discussion topics
            - Related workflow processes  
            - Integrated communication themes
            - Potential areas for template development
            """)
            
            # Show strongest relationships
            if len(keyword_pairs) > 0:
                strongest_pair = keyword_pairs[0]
                st.success(f"""
                🔗 **Strongest Keyword Relationship**: 
                "{strongest_pair['keyword1']}" ↔ "{strongest_pair['keyword2']}" 
                (appeared together {strongest_pair['cooccurrence_count']} times)
                """)
        
        # Temporal keyword analysis
        temporal_analysis = analysis.get('temporal_analysis', {})
        
        if temporal_analysis.get('monthly_keyword_trends'):
            st.subheader("📈 Keyword Trends Over Time")
            
            monthly_trends = temporal_analysis['monthly_keyword_trends']
            
            # Create trend visualization
            trend_data = []
            all_keywords = set()
            
            for month, keywords in monthly_trends.items():
                all_keywords.update(keywords.keys())
                for keyword, count in keywords.items():
                    trend_data.append({
                        'Month': month,
                        'Keyword': keyword,
                        'Count': count
                    })
            
            if trend_data:
                trend_df = pd.DataFrame(trend_data)
                
                # Show trends for top keywords
                top_keywords = trend_df.groupby('Keyword')['Count'].sum().nlargest(8).index
                trend_df_filtered = trend_df[trend_df['Keyword'].isin(top_keywords)]
                
                fig = px.line(
                    trend_df_filtered,
                    x='Month',
                    y='Count',
                    color='Keyword',
                    title="Keyword Usage Trends Over Time (Top 8 Keywords)",
                    markers=True
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            # Trend summary
            trend_summary = temporal_analysis.get('trend_summary', {})
            if trend_summary:
                col1, col2 = st.columns(2)
                
                with col1:
                    consistent_kw = trend_summary.get('consistent_keywords', [])
                    if consistent_kw:
                        st.write("**🔄 Consistently Used Keywords:**")
                        for kw in consistent_kw[:5]:
                            st.write(f"• {kw}")
                
                with col2:
                    emerging_kw = trend_summary.get('emerging_keywords', [])
                    if emerging_kw:
                        st.write("**🆕 Recently Emerging Keywords:**")
                        for kw in emerging_kw[:5]:
                            st.write(f"• {kw}")
    
    def _render_traditional_metrics_tab(self, analysis: Dict[str, Any]) -> None:
        """Render traditional quality metrics (maintaining existing functionality)"""
        st.header("📈 Traditional Quality Metrics")
        
        # Overview metrics (existing)
        col1, col2, col3, col4 = st.columns(4)
        
        quality_stats = analysis['quality_categories']['statistics']
        
        with col1:
            st.metric(
                "Average Substance Score",
                f"{analysis['avg_substance_score']:.2f}",
                help="Average keyword-based substance score across all emails"
            )
        
        with col2:
            high_quality_pct = quality_stats['high_quality']['percentage']
            st.metric(
                "High Quality Emails",
                f"{high_quality_pct:.1f}%",
                help="Emails with substantial business content (score ≥ 8)"
            )
        
        with col3:
            low_quality_pct = quality_stats['minimal_quality']['percentage']
            st.metric(
                "Low Substance Emails",
                f"{low_quality_pct:.1f}%", 
                help="Emails with minimal business substance (score < 1)"
            )
        
        with col4:
            total_emails = analysis['total_emails']
            st.metric(
                "Total Emails Analyzed",
                f"{total_emails:,}",
                help="Total number of emails in the analysis"
            )
        
        # Quality distribution (existing)
        st.subheader("📊 Quality Score Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Quality category pie chart
            category_data = pd.DataFrame({
                'Quality Level': ['High (≥8)', 'Medium (4-7)', 'Low (1-3)', 'Minimal (<1)'],
                'Count': [
                    quality_stats['high_quality']['count'],
                    quality_stats['medium_quality']['count'],
                    quality_stats['low_quality']['count'],
                    quality_stats['minimal_quality']['count']
                ]
            })
            
            # Filter out zero counts
            category_data = category_data[category_data['Count'] > 0]
            
            if len(category_data) > 0:
                fig = px.pie(
                    category_data,
                    names='Quality Level',
                    values='Count',
                    title="Email Quality Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Substance score histogram
            content_data = analysis['content_data']
            
            if len(content_data) > 0:
                fig = px.histogram(
                    content_data,
                    x='Substance_Score',
                    nbins=20,
                    title="Substance Score Distribution",
                    labels={'x': 'Substance Score', 'y': 'Number of Emails'}
                )
                
                # Add threshold lines
                fig.add_vline(x=8, line_dash="dash", line_color="green", 
                             annotation_text="High Quality")
                fig.add_vline(x=4, line_dash="dash", line_color="orange", 
                             annotation_text="Medium Quality")
                fig.add_vline(x=1, line_dash="dash", line_color="red", 
                             annotation_text="Low Quality")
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Direction comparison (existing)
        direction_analysis = analysis.get('direction_analysis', {})
        if direction_analysis:
            st.subheader("↔️ Quality by Email Direction")
            
            direction_data = []
            for direction, stats in direction_analysis.items():
                direction_data.append({
                    'Direction': direction,
                    'Count': stats['count'],
                    'Avg Substance Score': stats['avg_substance_score'],
                    'Avg Business Intensity': stats.get('avg_business_intensity', 0),
                    'Unique Keywords': stats.get('unique_keywords', 0),
                    'High Quality %': stats['high_quality_pct']
                })
            
            direction_df = pd.DataFrame(direction_data)
            
            if len(direction_df) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(
                        direction_df,
                        x='Direction',
                        y='Avg Substance Score',
                        title="Average Substance Score by Direction",
                        color='Avg Business Intensity',
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.dataframe(direction_df, use_container_width=True, hide_index=True)
        
        # Insights and recommendations (existing)
        st.subheader("💡 Insights & Recommendations")
        
        insights = self._generate_enhanced_insights(analysis)
        recommendations = self._generate_enhanced_recommendations(analysis)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Key Insights:**")
            for insight in insights:
                st.write(f"• {insight}")
        
        with col2:
            st.write("**🎯 Recommendations:**")
            for rec in recommendations:
                st.write(f"• {rec}")
    
    def _generate_enhanced_insights(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate enhanced insights based on keyword analysis"""
        insights = []
        
        # Keyword-based insights
        keyword_analysis = analysis['keyword_analysis']
        unique_keywords = keyword_analysis['unique_keywords_count']
        avg_keywords = keyword_analysis['avg_keywords_per_email']
        
        insights.append(f"Identified {unique_keywords} unique business keywords across communications")
        
        if avg_keywords >= 3:
            insights.append(f"Strong keyword density ({avg_keywords:.1f} keywords/email) indicates substantive discussions")
        elif avg_keywords >= 1.5:
            insights.append(f"Moderate keyword usage ({avg_keywords:.1f} keywords/email) shows mixed communication quality")
        else:
            insights.append(f"Low keyword density ({avg_keywords:.1f} keywords/email) suggests need for more substantive content")
        
        # Business intensity insights
        business_intensity = keyword_analysis.get('business_intensity', {})
        avg_intensity = business_intensity.get('avg_intensity', 0)
        
        if avg_intensity >= 2.5:
            insights.append("High business intensity indicates focus on strategic communications")
        elif avg_intensity >= 1.5:
            insights.append("Moderate business intensity shows balanced operational and strategic content")
        else:
            insights.append("Low business intensity suggests predominance of routine communications")
        
        # Co-occurrence insights
        cooccurrence = analysis.get('cooccurrence_analysis', {})
        if cooccurrence.get('keyword_pairs'):
            strong_pairs = len([p for p in cooccurrence['keyword_pairs'] if p['strength'] > 0.5])
            insights.append(f"Found {strong_pairs} strong keyword relationships indicating structured business processes")
        
        return insights
    
    def _generate_enhanced_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate enhanced recommendations based on keyword analysis"""
        recommendations = []
        
        keyword_analysis = analysis['keyword_analysis']
        avg_keywords = keyword_analysis['avg_keywords_per_email']
        unique_keywords = keyword_analysis['unique_keywords_count']
        
        # Keyword-based recommendations
        if avg_keywords < 2:
            recommendations.append("Increase use of specific business terminology in email communications")
        
        if unique_keywords < 50:
            recommendations.append("Expand business vocabulary to improve communication specificity")
        
        # Category-based recommendations
        cat_analysis = keyword_analysis.get('category_analysis', {})
        if cat_analysis:
            dominant_cat = cat_analysis.get('dominant_category')
            if dominant_cat == 'generic':
                recommendations.append("Focus on more specific action-oriented and commitment-based language")
        
        # Business context recommendations
        business_context = analysis.get('business_context', {})
        if business_context.get('business_themes'):
            theme_count = len(business_context['business_themes'])
            if theme_count < 3:
                recommendations.append("Diversify business communication themes for broader engagement")
        
        # Co-occurrence recommendations
        cooccurrence = analysis.get('cooccurrence_analysis', {})
        if cooccurrence.get('keyword_pairs'):
            recommendations.append("Leverage identified keyword relationships to create effective email templates")
        
        recommendations.extend([
            "Monitor keyword trends monthly to track communication evolution",
            "Develop keyword-based quality scoring for ongoing improvement",
            "Train team on effective business keyword usage patterns"
        ])
        
        return recommendations
    
    def get_report_summary(self) -> Dict[str, Any]:
        """Get enhanced summary metrics for dashboard overview"""
        analysis = self.generate_analysis()
        keyword_analysis = analysis['keyword_analysis']
        quality_stats = analysis['quality_categories']['statistics']
        
        return {
            'avg_substance_score': analysis['avg_substance_score'],
            'high_quality_percentage': quality_stats['high_quality']['percentage'],
            'unique_keywords_count': keyword_analysis['unique_keywords_count'],
            'avg_keywords_per_email': keyword_analysis['avg_keywords_per_email'],
            'business_intensity': keyword_analysis.get('business_intensity', {}).get('avg_intensity', 0),
            'content_quality_grade': self._calculate_enhanced_content_grade(analysis),
            'keyword_diversity_score': keyword_analysis.get('frequency_analysis', {}).get('keyword_diversity', 0)
        }
    
    def _calculate_enhanced_content_grade(self, analysis: Dict[str, Any]) -> str:
        """Calculate overall content quality grade with keyword weighting"""
        avg_score = analysis['avg_substance_score']
        quality_stats = analysis['quality_categories']['statistics']
        keyword_analysis = analysis['keyword_analysis']
        
        high_pct = quality_stats['high_quality']['percentage']
        keyword_density = keyword_analysis['avg_keywords_per_email']
        business_intensity = keyword_analysis.get('business_intensity', {}).get('avg_intensity', 0)
        
        # Enhanced weighted scoring
        score_component = min(100, (avg_score / 10) * 100)  # 40% weight
        quality_component = high_pct  # 30% weight
        keyword_component = min(100, keyword_density * 20)  # 20% weight
        intensity_component = min(100, business_intensity * 25)  # 10% weight
        
        overall_score = (
            score_component * 0.4 + 
            quality_component * 0.3 + 
            keyword_component * 0.2 + 
            intensity_component * 0.1
        )
        
        if overall_score >= 85:
            return "A+ - Exceptional"
        elif overall_score >= 80:
            return "A - Excellent"
        elif overall_score >= 70:
            return "B - Good"
        elif overall_score >= 60:
            return "C - Average"
        elif overall_score >= 50:
            return "D - Below Average"
        else:
            return "F - Poor"