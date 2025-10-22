"""
Data management system for CAF Dashboard
Handles data loading, validation, and type detection
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
import warnings
from datetime import datetime
import re

logger = logging.getLogger(__name__)


@dataclass
class ColumnInfo:
    """Information about a data column"""
    name: str
    dtype: str
    confidence: float
    sample_values: List[str]
    null_count: int
    unique_count: int
    is_temporal: bool = False


@dataclass
class DataInfo:
    """Information about loaded data"""
    file_path: str
    file_name: str
    file_format: str
    success: bool
    error_message: Optional[str] = None
    df: Optional[pd.DataFrame] = None
    columns_info: List[ColumnInfo] = None
    column_types: Dict[str, str] = None
    corrected_types: Dict[str, str] = None


class DataManager:
    """Centralized data management system"""
    
    def __init__(self):
        self.current_data: Optional[pd.DataFrame] = None
        self.column_types: Dict[str, str] = {}
        self.data_info: Optional[DataInfo] = None
        self.logger = logging.getLogger(f"{__name__}.DataManager")
        
        # Supported file formats
        self.supported_formats = {
            '.csv': 'CSV (Comma Separated Values)',
            '.xlsx': 'Excel (Excel 2007+)'
        }
    
    def load_file(self, file_path: str, **kwargs) -> DataInfo:
        """Load a single file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return DataInfo(
                file_path=str(file_path),
                file_name=file_path.name,
                file_format=file_path.suffix.lower(),
                success=False,
                error_message=f"File not found: {file_path}"
            )
        
        try:
            # Detect file format
            file_format = self._detect_file_format(file_path)
            
            # Load file based on format
            if file_format == '.csv':
                df = self._load_csv(file_path, **kwargs)
            elif file_format == '.xlsx':
                df = self._load_excel(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported format: {file_format}")
            
            if df is None or df.empty:
                return DataInfo(
                    file_path=str(file_path),
                    file_name=file_path.name,
                    file_format=file_format,
                    success=False,
                    error_message="File is empty or could not be loaded"
                )
            
            # Clean column names
            df = self._clean_column_names(df)
            
            # Detect column types
            columns_info = self._detect_columns_info(df)
            column_types = {col.name: col.dtype for col in columns_info}
            
            return DataInfo(
                file_path=str(file_path),
                file_name=file_path.name,
                file_format=file_format,
                success=True,
                df=df,
                columns_info=columns_info,
                column_types=column_types,
                corrected_types=column_types.copy()
            )
            
        except Exception as e:
            self.logger.error(f"Error loading file {file_path}: {str(e)}")
            return DataInfo(
                file_path=str(file_path),
                file_name=file_path.name,
                file_format=file_path.suffix.lower(),
                success=False,
                error_message=str(e)
            )
    
    def _detect_file_format(self, file_path: Path) -> str:
        """Detect file format"""
        extension = file_path.suffix.lower()
        
        if extension in self.supported_formats:
            return extension
        elif extension == '.txt':
            # Try to detect CSV by content
            return self._detect_csv_by_content(file_path)
        else:
            raise ValueError(f"Unsupported format: {extension}")
    
    def _detect_csv_by_content(self, file_path: Path) -> str:
        """Detect CSV by content"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
            
            if ',' in first_line and len(first_line.split(',')) > 2:
                return '.csv'
            else:
                raise ValueError("File does not appear to be a valid CSV")
        except:
            raise ValueError("Could not read file to detect format")
    
    def _load_csv(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load CSV file"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, **kwargs)
                    self.logger.info(f"Successfully loaded CSV with encoding: {encoding}")
                    return df
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, use utf-8 with errors='ignore'
            df = pd.read_csv(file_path, encoding='utf-8', errors='ignore', **kwargs)
            self.logger.warning("Loaded CSV with encoding errors ignored")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading CSV {file_path}: {str(e)}")
            raise
    
    def _load_excel(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load Excel file"""
        try:
            df = pd.read_excel(file_path, **kwargs)
            self.logger.info(f"Successfully loaded Excel file: {file_path}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading Excel {file_path}: {str(e)}")
            raise
    
    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean column names"""
        df = df.copy()
        
        # Remove extra whitespace
        df.columns = df.columns.str.strip()
        
        # Replace spaces with underscores
        df.columns = df.columns.str.replace(' ', '_')
        
        # Remove special characters
        df.columns = df.columns.str.replace(r'[^\w\s]', '', regex=True)
        
        # Ensure unique column names
        df.columns = self._make_unique_columns(df.columns)
        
        return df
    
    def _make_unique_columns(self, columns: pd.Index) -> pd.Index:
        """Make column names unique"""
        unique_columns = []
        seen = set()
        
        for col in columns:
            if col in seen:
                counter = 1
                while f"{col}_{counter}" in seen:
                    counter += 1
                unique_columns.append(f"{col}_{counter}")
                seen.add(f"{col}_{counter}")
            else:
                unique_columns.append(col)
                seen.add(col)
        
        return pd.Index(unique_columns)
    
    def _detect_columns_info(self, df: pd.DataFrame) -> List[ColumnInfo]:
        """Detect information about each column"""
        columns_info = []
        
        for col in df.columns:
            dtype, confidence = self._detect_column_type(df[col], col)
            
            col_info = ColumnInfo(
                name=col,
                dtype=dtype,
                confidence=confidence,
                sample_values=df[col].head(5).astype(str).tolist(),
                null_count=int(df[col].isna().sum()),
                unique_count=int(df[col].nunique()),
                is_temporal=self._is_temporal_column(df[col], col)
            )
            columns_info.append(col_info)
        
        return columns_info
    
    def _detect_column_type(self, series: pd.Series, column_name: str) -> Tuple[str, float]:
        """Detect column type with confidence"""
        # Check for temporal columns first
        if self._is_temporal_column(series, column_name):
            return 'datetime', 0.9
        
        # Check for numeric columns
        if self._is_numeric_column(series):
            return 'quantitative', 0.8
        
        # Default to qualitative
        return 'qualitative', 0.6
    
    def _is_temporal_column(self, series: pd.Series, column_name: str) -> bool:
        """Check if column is temporal"""
        # Check column name
        time_keywords = ['time', 'date', 'timestamp', 'fecha', 'hora', 'tiempo']
        if any(keyword in column_name.lower() for keyword in time_keywords):
            return True
        
        # Check data type
        try:
            pd.to_datetime(series, errors='raise')
            return True
        except (ValueError, TypeError):
            pass
        
        # Check sample values
        sample_values = series.dropna().head(10)
        if len(sample_values) > 0:
            try:
                pd.to_datetime(sample_values, errors='raise')
                return True
            except (ValueError, TypeError):
                pass
        
        return False
    
    def _is_numeric_column(self, series: pd.Series) -> bool:
        """Check if column is numeric"""
        try:
            pd.to_numeric(series, errors='raise')
            return True
        except (ValueError, TypeError):
            return False
    
    def set_current_data(self, data_info: DataInfo):
        """Set current data for analysis"""
        self.data_info = data_info
        self.current_data = data_info.df
        self.column_types = data_info.corrected_types or data_info.column_types
    
    def get_current_data(self) -> Optional[pd.DataFrame]:
        """Get current data"""
        return self.current_data
    
    def get_column_types(self) -> Dict[str, str]:
        """Get column types"""
        return self.column_types.copy()
    
    def update_column_type(self, column: str, new_type: str):
        """Update column type"""
        if self.column_types:
            self.column_types[column] = new_type
        if self.data_info and self.data_info.corrected_types:
            self.data_info.corrected_types[column] = new_type
    
    def cleanup(self):
        """Clean up data"""
        self.current_data = None
        self.column_types = {}
        self.data_info = None
        self.logger.info("Data manager cleaned up")
