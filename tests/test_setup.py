"""Test setup verification for HAM10000 QSPICE Pipeline."""

import pytest
import torch
import timm
from src.config import config_manager


def test_dependencies_available():
    """Test that all required dependencies are available."""
    # Test PyTorch
    assert torch.__version__ >= "2.0.0"
    
    # Test timm
    assert timm.__version__ >= "0.9.0"
    
    # Test EfficientNet availability
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=7)
    assert model is not None
    assert model.classifier.out_features == 7


def test_configuration_system():
    """Test that configuration system is working."""
    # Test configuration loading
    assert config_manager.model.architecture == "efficientnet_b0"
    assert config_manager.model.num_classes == 7
    assert config_manager.sram.cell_type == "6T"
    
    # Test configuration validation
    assert config_manager.data.test_size > 0
    assert config_manager.data.test_size < 1
    assert config_manager.model.batch_size > 0


def test_project_structure():
    """Test that project structure is correctly set up."""
    import os
    
    # Test directory structure
    required_dirs = [
        "src/data",
        "src/models", 
        "src/circuits",
        "src/analysis",
        "data/raw",
        "data/processed",
        "models/pretrained",
        "models/adapted",
        "models/sram",
        "circuits/qspice",
        "circuits/netlists",
        "results/performance",
        "results/plots",
        "results/reports",
        "tests",
        "config"
    ]
    
    for dir_path in required_dirs:
        assert os.path.exists(dir_path), f"Directory {dir_path} should exist"
    
    # Test configuration file
    assert os.path.exists("config/pipeline_config.yaml")
    
    # Test main files
    assert os.path.exists("src/main.py")
    assert os.path.exists("src/config.py")
    assert os.path.exists("requirements.txt")
    assert os.path.exists("setup.py")
    assert os.path.exists("README.md")


if __name__ == "__main__":
    pytest.main([__file__])