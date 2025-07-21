"""
Main Streamlit Application Entry Point
Meeting-Email Correlation Dashboard
"""

import streamlit as st
import pandas as pd
from pathlib import Path

# Import custom modules
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from logger import get_logger
from config import DashboardConfig

# Import data modules directly
from data.loader import ExcelDataLoader
from data.processor import DataProcessor
from dashboard.layout import DashboardLayout

# Initialize logger
logger = get_logger(__name__)


class ExcelDashboardApp:
    """
    Main Streamlit application class for Excel Dashboard
    """
    
    def __init__(self):
        """Initialize the dashboard application"""
        logger.info("Initializing Excel Dashboard Application")
        self.config = DashboardConfig()
        self.layout = DashboardLayout()
        self._initialize_session_state()
    
    def _initialize_session_state(self) -> None:
        """Initialize Streamlit session state variables"""
        logger.debug("Initializing session state variables")
        
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Home"
    
    def _configure_page(self) -> None:
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title=self.config.APP_TITLE,
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .report-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .metric-container {
            background-color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _load_data(self, uploaded_file=None) -> bool:
        """
        Load and process Excel data
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            bool: Success status of data loading
        """
        try:
            logger.info("Starting data loading process")
            
            if uploaded_file is not None:
                # Load from uploaded file
                loader = ExcelDataLoader()
                raw_data = loader.load_from_upload(uploaded_file)
                logger.info("Data loaded from uploaded file successfully")
            else:
                # Load from assets directory
                loader = ExcelDataLoader()
                excel_files = list(Path("assets/excel").glob("*.xlsx"))
                
                if not excel_files:
                    st.error("No Excel files found in assets/excel directory")
                    logger.error("No Excel files found in assets/excel directory")
                    return False
                
                # Use the first Excel file found
                raw_data = loader.load_from_file(excel_files[0])
                logger.info(f"Data loaded from file: {excel_files[0]}")
            
            # Process the data
            processor = DataProcessor()
            processed_data = processor.process_data(raw_data)
            
            # Store in session state
            st.session_state.processed_data = processed_data
            st.session_state.data_loaded = True
            
            logger.info(f"Data processing completed. Shape: {processed_data.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def _render_homepage(self) -> None:
        """Render the homepage with data loading interface"""
        logger.debug("Rendering homepage")
        
        st.markdown('<h1 class="main-header">📊 Excel Meeting-Email Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        ### Welcome to the Meeting-Email Correlation Analysis Dashboard
        
        This tool analyzes meeting engagement and email communication patterns by correlating 
        Outlook calendar meetings with corresponding email communications within a 48-hour window.
        """)
        
        # Data loading section
        st.markdown("### 📁 Data Loading")
        
        # Option 1: Upload file
        uploaded_file = st.file_uploader(
            "Upload Excel File",
            type=['xlsx', 'xls'],
            help="Upload your meeting-email correlation Excel file"
        )
        
        if uploaded_file is not None:
            if st.button("Load Uploaded Data", type="primary"):
                with st.spinner("Loading and processing data..."):
                    success = self._load_data(uploaded_file)
                    if success:
                        st.success("✅ Data loaded successfully!")
                        st.rerun()
        
        # Option 2: Load from assets
        st.markdown("**OR**")
        
        if st.button("Load from Assets Directory"):
            with st.spinner("Loading data from assets/excel directory..."):
                success = self._load_data()
                if success:
                    st.success("✅ Data loaded successfully!")
                    st.rerun()
        
        # Display data info if loaded
        if st.session_state.data_loaded:
            self._display_data_overview()
    
    def _display_data_overview(self) -> None:
        """Display overview of loaded data"""
        logger.debug("Displaying data overview")
        
        data = st.session_state.processed_data
        
        st.markdown("### 📈 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(data):,}")
        
        with col2:
            unique_meetings = data['Meeting Subject'].nunique()
            st.metric("Unique Meetings", f"{unique_meetings:,}")
        
        with col3:
            unique_emails = data['Email Subject'].nunique()
            st.metric("Unique Emails", f"{unique_emails:,}")
        
        with col4:
            date_range = (data['Meeting Time (BST/GMT)'].max() - data['Meeting Time (BST/GMT)'].min()).days
            st.metric("Date Range (Days)", f"{date_range:,}")
        
        # Sample data preview
        st.markdown("### 🔍 Data Preview")
        st.dataframe(
            data.head(10),
            use_container_width=True,
            hide_index=True
        )
    
    def run(self) -> None:
        """Main application runner"""
        logger.info("Starting Excel Dashboard Application")
        
        # Configure page
        self._configure_page()
        
        # Render sidebar navigation
        selected_page = self.layout.render_sidebar()
        
        # Route to appropriate page
        if selected_page == "Home":
            self._render_homepage()
        elif st.session_state.data_loaded:
            # Route to report pages
            self.layout.render_page(selected_page, st.session_state.processed_data)
        else:
            st.warning("⚠️ Please load data first from the Home page")
            st.stop()


def main():
    """Application entry point"""
    try:
        app = ExcelDashboardApp()
        app.run()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"Application error: {str(e)}")


if __name__ == "__main__":
    main()