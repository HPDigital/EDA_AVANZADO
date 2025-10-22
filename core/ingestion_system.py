"""
Data ingestion system for CAF Dashboard
Handles multi-step data loading and processing
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

from .data_manager import DataManager, DataInfo, ColumnInfo
from .memory_manager import MemoryManager

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


@dataclass
class IngestionStep:
    """Represents a step in the ingestion process"""
    step_number: int
    name: str
    description: str
    completed: bool = False
    data: Optional[Any] = None
    error_message: Optional[str] = None


class IngestionSystem:
    """Multi-step data ingestion system"""
    
    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        self.data_manager = DataManager()
        self.memory_manager = memory_manager or MemoryManager()
        self.logger = logging.getLogger(f"{__name__}.IngestionSystem")
        
        # Ingestion state
        self.steps: List[IngestionStep] = []
        self.current_step = 0
        self.loaded_files: List[DataInfo] = []
        self.unified_data: Optional[pd.DataFrame] = None
        
        # Initialize steps
        self._initialize_steps()
    
    def _initialize_steps(self):
        """Initialize ingestion steps"""
        self.steps = [
            IngestionStep(1, "file_selection", "Select files to load"),
            IngestionStep(2, "data_preview", "Preview and correct data types"),
            IngestionStep(3, "data_unification", "Configure data unification")
        ]
    
    def step1_select_files(self, file_paths: List[str]) -> List[DataInfo]:
        """Step 1: Select and load files"""
        self.logger.info(f"Step 1: Loading {len(file_paths)} files")
        
        loaded_files = []
        for file_path in file_paths:
            try:
                data_info = self.data_manager.load_file(file_path)
                loaded_files.append(data_info)
                
                if data_info.success:
                    self.logger.info(f"Successfully loaded: {data_info.file_name}")
                else:
                    self.logger.error(f"Failed to load {data_info.file_name}: {data_info.error_message}")
                    
            except Exception as e:
                self.logger.error(f"Error loading {file_path}: {str(e)}")
                loaded_files.append(DataInfo(
                    file_path=file_path,
                    file_name=Path(file_path).name,
                    file_format=Path(file_path).suffix.lower(),
                    success=False,
                    error_message=str(e)
                ))
        
        self.loaded_files = loaded_files
        self.steps[0].completed = True
        self.steps[0].data = loaded_files
        
        # Cleanup memory if needed
        self.memory_manager.monitor_memory()
        
        return loaded_files
    
    def step2_preview_data(self, file_index: int) -> Optional[DataInfo]:
        """Step 2: Preview data for a specific file"""
        if not (0 <= file_index < len(self.loaded_files)):
            self.logger.error(f"Invalid file index: {file_index}")
            return None
        
        file_info = self.loaded_files[file_index]
        if not file_info.success:
            self.logger.error(f"File {file_info.file_name} was not loaded successfully")
            return None
        
        self.logger.info(f"Step 2: Previewing data for {file_info.file_name}")
        return file_info
    
    def step2_update_column_type(self, file_index: int, column: str, new_type: str):
        """Step 2: Update column type for a specific file"""
        if not (0 <= file_index < len(self.loaded_files)):
            self.logger.error(f"Invalid file index: {file_index}")
            return
        
        file_info = self.loaded_files[file_index]
        if not file_info.success:
            self.logger.error(f"File {file_info.file_name} was not loaded successfully")
            return
        
        # Update column type
        if file_info.corrected_types is None:
            file_info.corrected_types = file_info.column_types.copy()
        
        file_info.corrected_types[column] = new_type
        self.logger.info(f"Updated column {column} type to {new_type} for {file_info.file_name}")
    
    def step3_unify_data(self, pivot_columns: Optional[Dict[int, str]] = None) -> pd.DataFrame:
        """Step 3: Unify data from multiple files"""
        self.logger.info("Step 3: Unifying data")
        
        successful_files = [f for f in self.loaded_files if f.success]
        
        if len(successful_files) == 0:
            self.logger.error("No successful files to unify")
            return pd.DataFrame()
        
        if len(successful_files) == 1:
            # Single file - no unification needed
            file_info = successful_files[0]
            unified_df = file_info.df.copy()
            unified_df['_source_file'] = file_info.file_name
            unified_df['_file_index'] = 0
            unified_df['_file_format'] = file_info.file_format
            
            # Add type metadata
            for col, col_type in (file_info.corrected_types or file_info.column_types).items():
                if col in unified_df.columns:
                    unified_df[f'_type_{col}'] = col_type
            
            self.unified_data = unified_df
            self.steps[2].completed = True
            self.steps[2].data = unified_df
            
            self.logger.info(f"Single file unified: {len(unified_df)} rows, {len(unified_df.columns)} columns")
            return unified_df
        
        # Multiple files - unify using pivot columns
        if not pivot_columns:
            self.logger.error("Pivot columns required for multiple files")
            return pd.DataFrame()
        
        unified_dfs = []
        
        for i, file_info in enumerate(successful_files):
            if i not in pivot_columns:
                self.logger.warning(f"No pivot column specified for file {i}: {file_info.file_name}")
                continue
            
            pivot_col = pivot_columns[i]
            if pivot_col not in file_info.df.columns:
                self.logger.warning(f"Pivot column {pivot_col} not found in {file_info.file_name}")
                continue
            
            df = file_info.df.copy()
            df['_source_file'] = file_info.file_name
            df['_file_index'] = i
            df['_file_format'] = file_info.file_format
            
            # Add type metadata
            for col, col_type in (file_info.corrected_types or file_info.column_types).items():
                if col in df.columns:
                    df[f'_type_{col}'] = col_type
            
            unified_dfs.append(df)
        
        if not unified_dfs:
            self.logger.error("No valid files to unify")
            return pd.DataFrame()
        
        # Unify dataframes
        try:
            # Find common pivot column
            common_pivot = None
            for df in unified_dfs:
                for col in df.columns:
                    if col.startswith('_type_') or col in ['_source_file', '_file_index', '_file_format']:
                        continue
                    if all(col in other_df.columns for other_df in unified_dfs):
                        common_pivot = col
                        break
                if common_pivot:
                    break
            
            if common_pivot:
                # Merge on common column
                unified_df = unified_dfs[0]
                for df in unified_dfs[1:]:
                    unified_df = pd.merge(unified_df, df, on=common_pivot, how='outer', suffixes=('', '_dup'))
            else:
                # Concatenate without merging
                unified_df = pd.concat(unified_dfs, ignore_index=True)
            
            self.unified_data = unified_df
            self.steps[2].completed = True
            self.steps[2].data = unified_df
            
            self.logger.info(f"Data unified: {len(unified_df)} rows, {len(unified_df.columns)} columns")
            return unified_df
            
        except Exception as e:
            self.logger.error(f"Error unifying data: {str(e)}")
            return pd.DataFrame()
    
    def get_unified_data(self) -> Optional[pd.DataFrame]:
        """Get unified data"""
        return self.unified_data
    
    def get_loaded_files(self) -> List[DataInfo]:
        """Get loaded files information"""
        return self.loaded_files.copy()
    
    def get_step_status(self, step_number: int) -> Optional[IngestionStep]:
        """Get status of a specific step"""
        if 1 <= step_number <= len(self.steps):
            return self.steps[step_number - 1]
        return None
    
    def is_step_completed(self, step_number: int) -> bool:
        """Check if a step is completed"""
        step = self.get_step_status(step_number)
        return step.completed if step else False
    
    def reset(self):
        """Reset ingestion system"""
        self.logger.info("Resetting ingestion system")
        
        # Clean up data
        for file_info in self.loaded_files:
            if file_info.df is not None:
                del file_info.df
        
        self.loaded_files = []
        self.unified_data = None
        
        # Reset steps
        for step in self.steps:
            step.completed = False
            step.data = None
            step.error_message = None
        
        self.current_step = 0
        
        # Force memory cleanup
        self.memory_manager.force_cleanup()
        
        self.logger.info("Ingestion system reset completed")
    
    def cleanup(self):
        """Clean up ingestion system"""
        self.reset()
        self.data_manager.cleanup()
        self.logger.info("Ingestion system cleaned up")
