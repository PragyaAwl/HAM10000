"""Example of how PyYAML will be used in HAM10000 QSPICE Pipeline."""

import yaml
from pathlib import Path
import json

# Example 1: Loading your pipeline configuration (Task 1)
def load_pipeline_config():
    """Load the main pipeline configuration file."""
    config_path = Path("config/pipeline_config.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Loaded pipeline configuration:")
    print(f"  Model architecture: {config['model']['architecture']}")
    print(f"  Batch size: {config['model']['batch_size']}")
    print(f"  SRAM cell type: {config['sram']['cell_type']}")
    print(f"  Supply voltage: {config['sram']['supply_voltage']}V")
    
    return config

# Example 2: Creating experiment-specific configurations
def create_experiment_configs():
    """Create different configuration files for various experiments."""
    
    # Base configuration
    base_config = {
        'experiment_name': 'baseline',
        'model': {
            'architecture': 'efficientnet_b0',
            'batch_size': 16,
            'learning_rate': 0.0001,
            'epochs': 50
        },
        'sram': {
            'cell_type': '6T',
            'supply_voltage': 1.2,
            'transistor_width': 0.5
        }
    }
    
    # Experiment 1: High accuracy with EfficientNet-B4
    high_accuracy_config = base_config.copy()
    high_accuracy_config.update({
        'experiment_name': 'high_accuracy',
        'model': {
            'architecture': 'efficientnet_b4',
            'batch_size': 8,
            'learning_rate': 0.00005,
            'epochs': 100,
            'input_size': [380, 380]
        }
    })
    
    # Experiment 2: Low power SRAM
    low_power_config = base_config.copy()
    low_power_config.update({
        'experiment_name': 'low_power',
        'sram': {
            'cell_type': '8T',
            'supply_voltage': 0.8,  # Lower voltage for power savings
            'transistor_width': 0.3
        }
    })
    
    # Experiment 3: High precision SRAM
    high_precision_config = base_config.copy()
    high_precision_config.update({
        'experiment_name': 'high_precision',
        'sram': {
            'cell_type': '6T',
            'supply_voltage': 1.5,  # Higher voltage for better precision
            'transistor_width': 0.8,
            'bit_line_capacitance': 5e-16  # Lower capacitance
        }
    })
    
    # Save experiment configurations
    experiments_dir = Path("experiments")
    experiments_dir.mkdir(exist_ok=True)
    
    configs = [
        (base_config, "baseline_config.yaml"),
        (high_accuracy_config, "high_accuracy_config.yaml"),
        (low_power_config, "low_power_config.yaml"),
        (high_precision_config, "high_precision_config.yaml")
    ]
    
    for config, filename in configs:
        config_path = experiments_dir / filename
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        print(f"Created: {config_path}")
    
    return configs

# Example 3: Dynamic configuration updates
def update_config_for_experiment(config_file, updates):
    """Dynamically update configuration for experiments."""
    # Load existing config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply updates (nested dictionary update)
    def deep_update(base_dict, update_dict):
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict:
                deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    deep_update(config, updates)
    
    # Save updated config
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Updated {config_file} with: {updates}")
    return config

# Example 4: Configuration validation
def validate_config(config):
    """Validate configuration parameters."""
    errors = []
    
    # Validate model configuration
    if 'model' not in config:
        errors.append("Missing 'model' section")
    else:
        model_config = config['model']
        
        # Check required fields
        required_fields = ['architecture', 'batch_size', 'learning_rate']
        for field in required_fields:
            if field not in model_config:
                errors.append(f"Missing model.{field}")
        
        # Check value ranges
        if 'batch_size' in model_config:
            if not isinstance(model_config['batch_size'], int) or model_config['batch_size'] <= 0:
                errors.append("model.batch_size must be a positive integer")
        
        if 'learning_rate' in model_config:
            lr = model_config['learning_rate']
            if not isinstance(lr, (int, float)) or lr <= 0 or lr > 1:
                errors.append("model.learning_rate must be between 0 and 1")
    
    # Validate SRAM configuration
    if 'sram' not in config:
        errors.append("Missing 'sram' section")
    else:
        sram_config = config['sram']
        
        # Check voltage ranges
        if 'supply_voltage' in sram_config:
            voltage = sram_config['supply_voltage']
            if not isinstance(voltage, (int, float)) or voltage <= 0 or voltage > 3.0:
                errors.append("sram.supply_voltage must be between 0 and 3.0 volts")
        
        # Check cell type
        if 'cell_type' in sram_config:
            if sram_config['cell_type'] not in ['6T', '8T']:
                errors.append("sram.cell_type must be '6T' or '8T'")
    
    return len(errors) == 0, errors

# Example 5: Configuration comparison
def compare_configs(config1_path, config2_path):
    """Compare two configuration files and show differences."""
    with open(config1_path, 'r') as f:
        config1 = yaml.safe_load(f)
    
    with open(config2_path, 'r') as f:
        config2 = yaml.safe_load(f)
    
    def find_differences(dict1, dict2, path=""):
        differences = []
        
        # Check all keys in dict1
        for key in dict1:
            current_path = f"{path}.{key}" if path else key
            
            if key not in dict2:
                differences.append(f"Key '{current_path}' only in config1")
            elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                differences.extend(find_differences(dict1[key], dict2[key], current_path))
            elif dict1[key] != dict2[key]:
                differences.append(f"'{current_path}': {dict1[key]} vs {dict2[key]}")
        
        # Check keys only in dict2
        for key in dict2:
            if key not in dict1:
                current_path = f"{path}.{key}" if path else key
                differences.append(f"Key '{current_path}' only in config2")
        
        return differences
    
    differences = find_differences(config1, config2)
    
    print(f"Comparing {config1_path} vs {config2_path}:")
    if differences:
        for diff in differences:
            print(f"  - {diff}")
    else:
        print("  No differences found")
    
    return differences

# Example 6: Configuration templates
def create_config_template():
    """Create a configuration template with comments."""
    template = """# HAM10000 QSPICE Pipeline Configuration Template

# Experiment Information
experiment_name: "my_experiment"
description: "Description of this experiment"
author: "Your Name"
date: "2024-01-01"

# Data Configuration
data:
  raw_data_path: "data/raw"
  processed_data_path: "data/processed"
  test_size: 0.2          # Fraction for test set (0.1-0.3)
  validation_size: 0.1    # Fraction for validation set (0.1-0.2)
  random_seed: 42         # For reproducible splits

# Model Configuration
model:
  architecture: "efficientnet_b0"  # Options: efficientnet_b0, efficientnet_b4
  pretrained: true                 # Use ImageNet pre-trained weights
  num_classes: 7                   # HAM10000 has 7 skin lesion classes
  input_size: [224, 224]           # Input image size [height, width]
  batch_size: 16                   # Batch size for training
  learning_rate: 0.0001            # Learning rate (0.00001-0.01)
  epochs: 50                       # Number of training epochs
  early_stopping_patience: 10      # Stop if no improvement for N epochs

# SRAM Circuit Configuration
sram:
  cell_type: "6T"                  # Options: "6T", "8T"
  supply_voltage: 1.2              # Supply voltage in volts (0.8-1.8)
  transistor_width: 0.5            # Transistor width in micrometers
  transistor_length: 0.18          # Transistor length in micrometers
  
# QSPICE Simulation Configuration
qspice:
  simulation_time: 1e-6            # Simulation time in seconds
  time_step: 1e-12                 # Time step in seconds
  temperature: 27                  # Temperature in Celsius

# Analysis Configuration
analysis:
  metrics: ["accuracy", "precision", "recall", "f1_score"]
  plot_formats: ["png", "pdf"]
  report_format: "pdf"
"""
    
    template_path = Path("config_template.yaml")
    with open(template_path, 'w') as f:
        f.write(template)
    
    print(f"Created configuration template: {template_path}")
    return template_path

# Example 7: Converting between YAML and JSON
def convert_yaml_to_json(yaml_file, json_file):
    """Convert YAML configuration to JSON format."""
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    
    with open(json_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Converted {yaml_file} to {json_file}")

if __name__ == "__main__":
    print("=== HAM10000 PyYAML Usage Examples ===\n")
    
    print("1. Loading Pipeline Configuration:")
    try:
        config = load_pipeline_config()
    except FileNotFoundError:
        print("  Pipeline config not found (run from HAM10000 directory)")
    
    print("\n2. Creating Experiment Configurations:")
    create_experiment_configs()
    
    print("\n3. Configuration Template:")
    create_config_template()
    
    print("\n4. Configuration Validation:")
    sample_config = {
        'model': {
            'architecture': 'efficientnet_b0',
            'batch_size': 16,
            'learning_rate': 0.0001
        },
        'sram': {
            'cell_type': '6T',
            'supply_voltage': 1.2
        }
    }
    is_valid, errors = validate_config(sample_config)
    print(f"  Sample config valid: {is_valid}")
    if errors:
        for error in errors:
            print(f"    Error: {error}")
    
    print("\n5. YAML to JSON Conversion:")
    if Path("config_template.yaml").exists():
        convert_yaml_to_json("config_template.yaml", "config_template.json")
        print("  ✅ Conversion completed")