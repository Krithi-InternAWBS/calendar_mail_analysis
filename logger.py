"""
Logging Configuration for Excel Dashboard Application
Implements comprehensive logging with lazy formatting
"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Optional

from config import DashboardConfig


class LazyFormatter(logging.Formatter):
    """
    Custom formatter that implements lazy string formatting for better performance
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with lazy evaluation
        
        Args:
            record: LogRecord instance
            
        Returns:
            str: Formatted log message
        """
        # Handle lazy formatting if message is a tuple
        if isinstance(record.msg, tuple) and len(record.msg) >= 2:
            template, args = record.msg[0], record.msg[1:]
            record.msg = template
            record.args = args
        
        return super().format(record)


class DashboardLogger:
    """
    Centralized logging configuration class for the dashboard application
    """
    
    def __init__(self):
        """Initialize logging configuration"""
        self.config = DashboardConfig()
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Configure logging settings and handlers"""
        
        # Create logs directory if it doesn't exist
        self.config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.config.LOG_LEVEL)
        
        # Clear existing handlers to avoid duplicates
        root_logger.handlers.clear()
        
        # Create formatters
        self._setup_formatters()
        
        # Create and configure handlers
        self._setup_file_handler()
        self._setup_console_handler()
        self._setup_error_handler()
    
    def _setup_formatters(self) -> None:
        """Setup log formatters"""
        
        # Detailed formatter for file logs
        self.detailed_formatter = LazyFormatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt=self.config.LOG_DATE_FORMAT
        )
        
        # Simple formatter for console logs
        self.simple_formatter = LazyFormatter(
            fmt="%(levelname)s | %(name)s | %(message)s",
            datefmt=self.config.LOG_DATE_FORMAT
        )
        
        # Error formatter with more detail
        self.error_formatter = LazyFormatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(pathname)s:%(lineno)d | %(funcName)s | %(message)s",
            datefmt=self.config.LOG_DATE_FORMAT
        )
    
    def _setup_file_handler(self) -> None:
        """Setup rotating file handler for general logs"""
        
        log_file = self.config.LOGS_DIR / "dashboard.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(self.detailed_formatter)
        
        # Add to root logger
        logging.getLogger().addHandler(file_handler)
    
    def _setup_console_handler(self) -> None:
        """Setup console handler for development"""
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self.simple_formatter)
        
        # Add to root logger
        logging.getLogger().addHandler(console_handler)
    
    def _setup_error_handler(self) -> None:
        """Setup separate handler for errors and warnings"""
        
        error_log_file = self.config.LOGS_DIR / "errors.log"
        
        error_handler = logging.handlers.RotatingFileHandler(
            filename=error_log_file,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,
            encoding='utf-8'
        )
        
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(self.error_formatter)
        
        # Add to root logger
        logging.getLogger().addHandler(error_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger instance with the specified name
        
        Args:
            name: Logger name (typically __name__)
            
        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger(name)
        
        # Add context information
        logger.info("Logger initialized for module: %s", name)
        
        return logger


class LoggerMixin:
    """
    Mixin class to add logging capabilities to any class
    """
    
    @property
    def logger(self) -> logging.Logger:
        """
        Get logger instance for the current class
        
        Returns:
            logging.Logger: Logger instance
        """
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger
    
    def log_method_entry(self, method_name: str, **kwargs) -> None:
        """
        Log method entry with parameters
        
        Args:
            method_name: Name of the method
            **kwargs: Method parameters to log
        """
        if kwargs:
            self.logger.debug("Entering %s with parameters: %s", method_name, kwargs)
        else:
            self.logger.debug("Entering %s", method_name)
    
    def log_method_exit(self, method_name: str, result: Optional[any] = None) -> None:
        """
        Log method exit with optional result
        
        Args:
            method_name: Name of the method
            result: Method result to log (optional)
        """
        if result is not None:
            self.logger.debug("Exiting %s with result type: %s", method_name, type(result).__name__)
        else:
            self.logger.debug("Exiting %s", method_name)


# Global logger instance
_dashboard_logger = None


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    global _dashboard_logger
    
    if _dashboard_logger is None:
        _dashboard_logger = DashboardLogger()
    
    return _dashboard_logger.get_logger(name)


def log_function_call(func):
    """
    Decorator to automatically log function calls with lazy formatting
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        # Log function entry
        logger.debug("Calling function: %s with args: %s, kwargs: %s", 
                    func.__name__, len(args), len(kwargs))
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Log successful exit
            logger.debug("Function %s completed successfully", func.__name__)
            
            return result
            
        except Exception as e:
            # Log error
            logger.error("Function %s failed with error: %s", func.__name__, str(e))
            raise
    
    return wrapper


def log_performance(func):
    """
    Decorator to log function performance metrics
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function with performance logging
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        start_time = datetime.now()
        
        try:
            result = func(*args, **kwargs)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.info("Function %s executed in %.3f seconds", func.__name__, execution_time)
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error("Function %s failed after %.3f seconds with error: %s", 
                        func.__name__, execution_time, str(e))
            raise
    
    return wrapper


# Initialize logging on module import
try:
    _dashboard_logger = DashboardLogger()
    module_logger = get_logger(__name__)
    module_logger.info("Logging system initialized successfully")
except Exception as e:
    print(f"Failed to initialize logging: {e}")
    raise