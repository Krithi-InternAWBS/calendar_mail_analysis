"""
Configuration Settings for Excel Dashboard Application
"""

import logging
from pathlib import Path
from typing import List, Dict, Any


class DashboardConfig:
    """
    Configuration class containing all dashboard settings and constants
    """
    
    # Application Settings
    APP_TITLE = "Excel Meeting-Email Correlation Dashboard"
    APP_VERSION = "1.0.0"
    APP_DESCRIPTION = "Forensic analysis tool for meeting engagement validation"
    
    # File Paths
    ASSETS_DIR = Path("assets")
    EXCEL_DIR = ASSETS_DIR / "excel"
    CRM_DIR = ASSETS_DIR / "crm"
    LOGS_DIR = Path("logs")
    
    # Data Processing Settings
    TIME_WINDOW_HOURS = 48  # 48-hour window for email-meeting correlation
    
    # Expected Excel Columns
    REQUIRED_COLUMNS = [
        "Meeting Subject",
        "Organizer",
        "Attendees", 
        "Email Subject",
        "Mail Sender",
        "Mail Receiver",
        "Email Body Content",
        "Direction",
        "Email Time (BST/GMT)",
        "Meeting Time (BST/GMT)",
        "Time Delta (hrs)"
    ]
    
    # Date/Time Format Settings
    DATETIME_FORMATS = [
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%m/%d/%Y %H:%M",
        "%Y-%m-%d %H:%M",
        "%d-%m-%Y %H:%M:%S"
    ]
    
    # NLP Keywords for Content Quality Analysis
    ENGAGEMENT_KEYWORDS = {
        'high_value': [
            'next steps',
            'proposal',
            'follow-up',
            'confirmation',
            'action item',
            'action items',
            'schedule',
            'attached',
            'agreement',
            'contract',
            'decision',
            'timeline',
            'deliverable',
            'milestone',
            'budget',
            'investment',
            'roi',
            'revenue',
            'sales',
            'opportunity'
        ],
        'medium_value': [
            'meeting',
            'discussion',
            'review',
            'update',
            'status',
            'progress',
            'feedback',
            'question',
            'clarification',
            'information',
            'details',
            'requirements'
        ],
        'low_value': [
            'thank you',
            'thanks',
            'received',
            'noted',
            'okay',
            'got it',
            'understood',
            'will do',
            'sounds good',
            'no problem'
        ]
    }
    
    # Response Time Categories (in hours)
    RESPONSE_TIME_CATEGORIES = {
        'immediate': 1,      # Within 1 hour
        'quick': 4,          # Within 4 hours  
        'same_day': 12,      # Within 12 hours
        'next_day': 24,      # Within 24 hours
        'two_days': 48,      # Within 48 hours
        'slow': float('inf') # More than 48 hours
    }
    
    # Logging Configuration
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    # Streamlit Page Configuration
    PAGE_CONFIG = {
        'page_title': APP_TITLE,
        'page_icon': '📊',
        'layout': 'wide',
        'initial_sidebar_state': 'expanded'
    }
    
    # Chart Color Schemes
    COLOR_SCHEMES = {
        'primary': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
        'engagement': ['#e74c3c', '#f39c12', '#27ae60', '#3498db', '#9b59b6'],
        'response_time': ['#27ae60', '#f39c12', '#e74c3c', '#95a5a6'],
        'trend': ['#3498db', '#e74c3c']
    }
    
    # Report Configurations
    REPORT_CONFIGS = {
        'engagement': {
            'title': 'Meeting Engagement Validation',
            'description': 'Identifies meetings with zero email correspondence',
            'icon': '🎯',
            'color': '#e74c3c'
        },
        'responsiveness': {
            'title': 'Responsiveness & Follow-Up Timeliness',
            'description': 'Measures email response delays and patterns',
            'icon': '⚡',
            'color': '#f39c12'
        },
        'client_mapping': {
            'title': 'Client Communication Mapping',
            'description': 'Maps client stakeholder engagement patterns',
            'icon': '🗺️',
            'color': '#3498db'
        },
        'content_quality': {
            'title': 'Engagement Quality Summary',
            'description': 'Analyzes email content for business substance',
            'icon': '📝',
            'color': '#27ae60'
        },
        'trends': {
            'title': 'Monthly Activity Trend Report',
            'description': 'Tracks engagement patterns over time',
            'icon': '📈',
            'color': '#9b59b6'
        }
    }
    
    # Validation Rules
    VALIDATION_RULES = {
        'min_records': 10,           # Minimum records for analysis
        'max_time_delta': 168,       # Maximum time delta in hours (7 days)
        'required_fill_rate': 0.8,   # Minimum fill rate for required columns
        'email_format_regex': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    }
    
    @classmethod
    def get_keyword_score_weights(cls) -> Dict[str, int]:
        """
        Get scoring weights for different keyword categories
        
        Returns:
            Dict[str, int]: Keyword category weights
        """
        return {
            'high_value': 3,
            'medium_value': 2, 
            'low_value': 1
        }
    
    @classmethod
    def get_all_keywords(cls) -> List[str]:
        """
        Get all engagement keywords as a flat list
        
        Returns:
            List[str]: All keywords combined
        """
        all_keywords = []
        for category_keywords in cls.ENGAGEMENT_KEYWORDS.values():
            all_keywords.extend(category_keywords)
        return all_keywords
    
    @classmethod
    def create_directories(cls) -> None:
        """Create necessary directories if they don't exist"""
        directories = [
            cls.ASSETS_DIR,
            cls.EXCEL_DIR,
            cls.CRM_DIR,
            cls.LOGS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_config(cls) -> bool:
        """
        Validate configuration settings
        
        Returns:
            bool: True if configuration is valid
        """
        try:
            # Check if required directories can be created
            cls.create_directories()
            
            # Validate keyword categories
            assert len(cls.ENGAGEMENT_KEYWORDS) > 0, "No engagement keywords defined"
            
            # Validate response time categories
            assert len(cls.RESPONSE_TIME_CATEGORIES) > 0, "No response time categories defined"
            
            # Validate required columns
            assert len(cls.REQUIRED_COLUMNS) > 0, "No required columns defined"
            
            return True
            
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {str(e)}")


# Initialize configuration on import
config = DashboardConfig()
config.create_directories()