"""
Core module for CAF Dashboard
Contains base classes and interfaces
"""
from .base_analyzer import BaseAnalyzer
from .data_manager import DataManager
from .config_manager import ConfigManager
from .memory_manager import MemoryManager

__all__ = [
    'BaseAnalyzer',
    'DataManager', 
    'ConfigManager',
    'MemoryManager'
]
