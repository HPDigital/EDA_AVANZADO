"""
Base analyzer class for all analysis modules
Provides common interface and functionality
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Base class for analysis results"""
    success: bool
    data: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None


class BaseAnalyzer(ABC):
    """Base class for all analyzers"""
    
    def __init__(self, name: str):
        self.name = name
        self.data: Optional[pd.DataFrame] = None
        self.column_types: Dict[str, str] = {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    def set_data(self, data: pd.DataFrame, column_types: Optional[Dict[str, str]] = None):
        """Set data for analysis"""
        self.data = data.copy()
        if column_types:
            self.column_types = column_types.copy()
        else:
            self.column_types = self._detect_column_types()
    
    def _detect_column_types(self) -> Dict[str, str]:
        """Detect column types automatically"""
        if self.data is None:
            return {}
        
        column_types = {}
        for col in self.data.columns:
            if self._is_numeric_column(col):
                column_types[col] = 'quantitative'
            elif self._is_datetime_column(col):
                column_types[col] = 'datetime'
            else:
                column_types[col] = 'qualitative'
        
        return column_types
    
    def _is_numeric_column(self, column: str) -> bool:
        """Check if column is numeric"""
        if self.data is None:
            return False
        
        try:
            pd.to_numeric(self.data[column], errors='raise')
            return True
        except (ValueError, TypeError):
            return False
    
    def _is_datetime_column(self, column: str) -> bool:
        """Check if column is datetime"""
        if self.data is None:
            return False
        
        try:
            pd.to_datetime(self.data[column], errors='raise')
            return True
        except (ValueError, TypeError):
            return False
    
    def _is_temporal_column(self, column: str) -> bool:
        """Check if column is temporal (datetime or time series)"""
        if self.data is None:
            return False
        
        # Check metadata first
        if column in self.column_types:
            return self.column_types[column] in ['datetime', 'timestamp']
        
        # Check if column name suggests time
        time_keywords = ['time', 'date', 'timestamp', 'fecha', 'hora', 'tiempo']
        if any(keyword in column.lower() for keyword in time_keywords):
            return True
        
        # Check data type
        return self._is_datetime_column(column)
    
    def get_numeric_columns(self) -> List[str]:
        """Get list of numeric columns"""
        return [col for col, dtype in self.column_types.items() 
                if dtype == 'quantitative']
    
    def get_categorical_columns(self) -> List[str]:
        """Get list of categorical columns"""
        return [col for col, dtype in self.column_types.items() 
                if dtype == 'qualitative']
    
    def get_temporal_columns(self) -> List[str]:
        """Get list of temporal columns"""
        return [col for col, dtype in self.column_types.items() 
                if dtype in ['datetime', 'timestamp']]
    
    def validate_data(self) -> bool:
        """Validate data before analysis"""
        if self.data is None or self.data.empty:
            self.logger.error("No data available for analysis")
            return False
        return True
    
    @abstractmethod
    def analyze(self, **kwargs) -> AnalysisResult:
        """Perform analysis - must be implemented by subclasses"""
        pass
    
    def cleanup(self):
        """Clean up resources"""
        self.data = None
        self.column_types = {}
        self.logger.info(f"{self.name} analyzer cleaned up")
