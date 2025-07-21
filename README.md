# Excel Meeting-Email Correlation Dashboard

A comprehensive Streamlit dashboard for analyzing meeting engagement and email communication patterns. This forensic analysis tool correlates Outlook calendar meetings with corresponding email communications within a 48-hour window.

## Project Overview

The objective of this dashboard is to:
- Extract and analyze calendar appointments from Excel exports
- Cross-reference emails sent/received to meeting attendees within 48-hour windows
- Generate 5 comprehensive visual reports for engagement validation
- Provide interactive dashboard for exploring communication patterns

## Features

### 5 Analytical Reports:
1. **Meeting Engagement Validation** - Identifies meetings with zero email correspondence
2. **Responsiveness & Follow-Up Timeliness** - Measures email response delays
3. **Client Communication Mapping** - Maps client stakeholder engagement patterns
4. **Engagement Quality Summary** - Analyzes email content for business substance
5. **Monthly Activity Trend Report** - Tracks engagement patterns over time

### Technical Features:
- **Modular OOP Design** with inheritance-based report classes
- **Comprehensive Logging** with lazy formatting throughout
- **Interactive Streamlit Interface** with multi-page navigation
- **Poetry Package Management** for dependency isolation
- **Plotly Visualizations** for professional charts

## Installation & Setup

### Prerequisites
- Python 3.9+
- Poetry (install from https://python-poetry.org/)

### Installation Steps

1. **Clone/Create project directory**:
```bash
mkdir excel-dashboard
cd excel-dashboard
```

2. **Install dependencies with Poetry**:
```bash
poetry install
```

3. **Activate virtual environment**:
```bash
poetry shell
```

4. **Prepare data directories**:
```bash
mkdir -p assets/excel assets/crm logs
```

5. **Place your Excel files** in the `assets/excel/` directory

## Usage

### Running the Dashboard
```bash
poetry run streamlit run app.py
```

### Data Format Expected
Your Excel file should contain these columns:
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

### Navigation
The dashboard provides:
- **Homepage** - Overview and data upload
- **Report 1** - Meeting Engagement Validation
- **Report 2** - Responsiveness Analysis  
- **Report 3** - Client Communication Mapping
- **Report 4** - Content Quality Analysis
- **Report 5** - Monthly Trend Analysis

## Project Structure

```
excel-dashboard/
├── pyproject.toml                 # Poetry configuration
├── README.md                      # This file
├── app.py                         # Streamlit entry point
├── config.py                      # Settings & configuration
├── logger.py                      # Logging setup
├── data/
│   ├── __init__.py
│   ├── loader.py                  # Excel data loading
│   └── processor.py               # Data preprocessing
├── reports/
│   ├── __init__.py
│   ├── base.py                    # Base report class
│   ├── engagement.py              # Report 1
│   ├── responsiveness.py          # Report 2
│   ├── client_mapping.py          # Report 3
│   ├── content_quality.py         # Report 4
│   └── trends.py                  # Report 5
├── dashboard/
│   ├── __init__.py
│   ├── layout.py                  # UI components
│   ├── charts.py                  # Visualization utilities
│   └── utils.py                   # Helper functions
├── assets/
│   ├── excel/                     # Input Excel files
│   └── crm/                       # CRM data files
└── logs/                          # Application logs
```

## Development

### Code Quality
```bash
poetry run black .                 # Code formatting
poetry run flake8 .               # Linting
poetry run isort .                # Import sorting
```

### Testing
```bash
poetry run pytest                 # Run tests
```

## Configuration

Modify `config.py` to adjust:
- NLP keywords for content analysis
- Time window thresholds
- Logging levels
- Chart styling preferences

## Logging

All operations are logged to `logs/` directory with:
- INFO level for normal operations
- WARNING for data quality issues
- ERROR for processing failures
- Lazy formatting for performance

## Contributing

1. Follow OOP principles with proper inheritance
2. Add comprehensive logging to all methods
3. Use lazy formatting for all log statements
4. Maintain modular structure
5. Update documentation for new features

## License

This project is for internal forensic analysis purposes.