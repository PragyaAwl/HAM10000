"""Configuration management for HAM10000 QSPICE Pipeline."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DataConfig:
    """Data processing configuration."""
    raw_data_path: str
    processed_data_path: str
    metadata_file: str
    image_dirs: list
    test_size: float
    validation_size: float
    random_seed: int


@dataclass
class ModelConfig:
    """Model configuration."""
    architecture: str
    pretrained: bool
    num_classes: int
    input_size: list
    batch_size: int
    learning_rate: float
    epochs: int
    early_stopping_patience: int


@dataclass
class PreprocessingConfig:
    """Image preprocessing configuration."""
    resize_size: list
    normalize_mean: list
    normalize_std: list


@dataclass
class SRAMConfig:
    """SRAM circuit configuration."""
    cell_type: str
    transistor_width: float
    transistor_length: float
    supply_voltage: float
    word_line_voltage: float
    bit_line_capacitance: float


@dataclass
class QSPICEConfig:
    """QSPICE simulation configuration."""
    simulation_time: float
    time_step: float
    temperature: float


@dataclass
class AnalysisConfig:
    """Analysis configuration."""
    metrics: list
    plot_formats: list
    report_format: str


@dataclass
class TestingConfig:
    """Testing configuration."""
    property_test_iterations: int
    unit_test_coverage_threshold: float


@dataclass
class PathsConfig:
    """Paths configuration."""
    models: Dict[str, str]
    circuits: Dict[str, str]
    results: Dict[str, str]


class ConfigManager:
    """Manages pipeline configuration loading and validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "pipeline_config.yaml"
        
        self.config_path = Path(config_path)
        self._config = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required_sections = ['data', 'model', 'preprocessing', 'sram', 'qspice', 
                           'analysis', 'testing', 'paths']
        
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate data configuration
        data_config = self._config['data']
        if data_config['test_size'] <= 0 or data_config['test_size'] >= 1:
            raise ValueError("test_size must be between 0 and 1")
        
        if data_config['validation_size'] <= 0 or data_config['validation_size'] >= 1:
            raise ValueError("validation_size must be between 0 and 1")
        
        # Validate model configuration
        model_config = self._config['model']
        if model_config['num_classes'] <= 0:
            raise ValueError("num_classes must be positive")
        
        if model_config['batch_size'] <= 0:
            raise ValueError("batch_size must be positive")
        
        # Validate SRAM configuration
        sram_config = self._config['sram']
        if sram_config['cell_type'] not in ['6T', '8T']:
            raise ValueError("cell_type must be '6T' or '8T'")
        
        if sram_config['supply_voltage'] <= 0:
            raise ValueError("supply_voltage must be positive")
    
    @property
    def data(self) -> DataConfig:
        """Get data configuration."""
        return DataConfig(**self._config['data'])
    
    @property
    def model(self) -> ModelConfig:
        """Get model configuration."""
        return ModelConfig(**self._config['model'])
    
    @property
    def preprocessing(self) -> PreprocessingConfig:
        """Get preprocessing configuration."""
        return PreprocessingConfig(**self._config['preprocessing'])
    
    @property
    def sram(self) -> SRAMConfig:
        """Get SRAM configuration."""
        return SRAMConfig(**self._config['sram'])
    
    @property
    def qspice(self) -> QSPICEConfig:
        """Get QSPICE configuration."""
        return QSPICEConfig(**self._config['qspice'])
    
    @property
    def analysis(self) -> AnalysisConfig:
        """Get analysis configuration."""
        return AnalysisConfig(**self._config['analysis'])
    
    @property
    def testing(self) -> TestingConfig:
        """Get testing configuration."""
        return TestingConfig(**self._config['testing'])
    
    @property
    def paths(self) -> PathsConfig:
        """Get paths configuration."""
        return PathsConfig(**self._config['paths'])
    
    def get_raw_config(self) -> Dict[str, Any]:
        """Get raw configuration dictionary."""
        return self._config.copy()
    
    def update_config(self, section: str, key: str, value: Any) -> None:
        """Update a configuration value.
        
        Args:
            section: Configuration section name
            key: Configuration key
            value: New value
        """
        if section not in self._config:
            raise ValueError(f"Unknown configuration section: {section}")
        
        self._config[section][key] = value
        self._validate_config()
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """Save current configuration to file.
        
        Args:
            output_path: Path to save configuration. If None, overwrites original.
        """
        if output_path is None:
            output_path = self.config_path
        
        with open(output_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)


# Global configuration instance
config_manager = ConfigManager()