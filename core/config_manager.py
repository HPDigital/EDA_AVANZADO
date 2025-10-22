"""
Configuration management for CAF Dashboard
Handles application settings and preferences
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """Application configuration"""
    # Window settings
    window_width: int = 1400
    window_height: int = 900
    window_title: str = "CAF Dashboard - EDA Avanzado"
    
    # Data settings
    default_data_dir: str = "data/raw"
    supported_formats: list = None
    
    # Analysis settings
    max_rows_preview: int = 50
    confidence_threshold: float = 0.7
    
    # UI settings
    theme: str = "clam"
    accent_color: str = "#4a90e2"
    bg_color: str = "#f0f0f0"
    fg_color: str = "#333333"
    
    # Memory settings
    max_memory_usage: float = 0.8  # 80% of available memory
    cleanup_threshold: int = 1000  # Cleanup after 1000 operations
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['.csv', '.xlsx']


class ConfigManager:
    """Manages application configuration"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = Path(config_file)
        self.config = AppConfig()
        self.logger = logging.getLogger(f"{__name__}.ConfigManager")
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Update config with loaded data
                for key, value in data.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                
                self.logger.info(f"Configuration loaded from {self.config_file}")
            except Exception as e:
                self.logger.error(f"Error loading config: {str(e)}")
                self._create_default_config()
        else:
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration file"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            self.save_config()
            self.logger.info(f"Default configuration created at {self.config_file}")
        except Exception as e:
            self.logger.error(f"Error creating default config: {str(e)}")
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.config), f, indent=2, ensure_ascii=False)
            self.logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            self.logger.error(f"Error saving config: {str(e)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return getattr(self.config, key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        if hasattr(self.config, key):
            setattr(self.config, key, value)
            self.save_config()
        else:
            self.logger.warning(f"Unknown configuration key: {key}")
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values"""
        return asdict(self.config)
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self.config = AppConfig()
        self.save_config()
        self.logger.info("Configuration reset to defaults")
