"""
Weight extraction utilities for HAM10000 QSPICE Pipeline.

This module provides functionality to extract model weights from trained neural networks,
organize them by layer type and size, and prepare them for SRAM storage simulation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
from collections import OrderedDict
import json

logger = logging.getLogger(__name__)


class WeightExtractor:
    """Utility class for extracting and organizing neural network weights."""
    
    def __init__(self):
        """Initialize the weight extractor."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def extract_all_weights(self, model: nn.Module) -> Dict[str, np.ndarray]:
        """
        Extract all parameters from a neural network model.
        
        Args:
            model: PyTorch model to extract weights from
            
        Returns:
            Dictionary mapping layer names to weight arrays
            
        Raises:
            ValueError: If model has no parameters
        """
        logger.info("Extracting all weights from neural network model...")
        
        # Ensure model is in evaluation mode
        model.eval()
        
        weights_dict = OrderedDict()
        total_params = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Convert to numpy array on CPU
                weight_array = param.detach().cpu().numpy()
                weights_dict[name] = weight_array
                total_params += param.numel()
                
                logger.debug(f"Extracted {name}: shape {weight_array.shape}, "
                           f"elements {param.numel()}")
        
        if not weights_dict:
            raise ValueError("Model has no trainable parameters")
        
        logger.info(f"Successfully extracted {len(weights_dict)} weight tensors "
                   f"with {total_params:,} total parameters")
        
        return weights_dict
    
    def organize_weights_by_layer_type(
        self, 
        weights_dict: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Organize weights by layer type for better SRAM mapping analysis.
        
        Args:
            weights_dict: Dictionary of layer names to weight arrays
            
        Returns:
            Dictionary organized by layer type (conv, linear, bn, etc.)
        """
        logger.info("Organizing weights by layer type...")
        
        organized_weights = {
            'convolutional': {},
            'linear': {},
            'batch_norm': {},
            'embedding': {},
            'other': {}
        }
        
        for layer_name, weights in weights_dict.items():
            # Determine layer type based on name patterns
            if any(keyword in layer_name.lower() for keyword in ['conv', 'depthwise', 'pointwise']):
                organized_weights['convolutional'][layer_name] = weights
            elif any(keyword in layer_name.lower() for keyword in ['linear', 'classifier', 'head', 'fc']):
                organized_weights['linear'][layer_name] = weights
            elif any(keyword in layer_name.lower() for keyword in ['bn', 'batch_norm', 'norm']):
                organized_weights['batch_norm'][layer_name] = weights
            elif 'embedding' in layer_name.lower():
                organized_weights['embedding'][layer_name] = weights
            else:
                organized_weights['other'][layer_name] = weights
        
        # Log organization summary
        for layer_type, layers in organized_weights.items():
            if layers:
                total_params = sum(w.size for w in layers.values())
                logger.info(f"{layer_type.title()}: {len(layers)} layers, "
                           f"{total_params:,} parameters")
        
        return organized_weights
    
    def get_weight_statistics(
        self, 
        weights_dict: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, Union[float, int, List[int]]]]:
        """
        Calculate comprehensive statistics for extracted weights.
        
        Args:
            weights_dict: Dictionary of layer names to weight arrays
            
        Returns:
            Dictionary with detailed statistics for each layer
        """
        logger.info("Calculating weight statistics...")
        
        statistics = {}
        
        for layer_name, weights in weights_dict.items():
            layer_stats = {
                'shape': list(weights.shape),
                'total_elements': int(weights.size),
                'dtype': str(weights.dtype),
                'mean': float(np.mean(weights)),
                'std': float(np.std(weights)),
                'min': float(np.min(weights)),
                'max': float(np.max(weights)),
                'abs_mean': float(np.mean(np.abs(weights))),
                'abs_max': float(np.max(np.abs(weights))),
                'zero_fraction': float(np.mean(weights == 0.0)),
                'memory_mb': float(weights.nbytes / (1024 * 1024))
            }
            
            # Add percentile information
            percentiles = [1, 5, 25, 50, 75, 95, 99]
            layer_stats['percentiles'] = {
                f'p{p}': float(np.percentile(weights, p)) for p in percentiles
            }
            
            statistics[layer_name] = layer_stats
        
        # Calculate overall statistics
        all_weights = np.concatenate([w.flatten() for w in weights_dict.values()])
        statistics['overall'] = {
            'total_parameters': int(len(all_weights)),
            'total_memory_mb': float(all_weights.nbytes / (1024 * 1024)),
            'global_mean': float(np.mean(all_weights)),
            'global_std': float(np.std(all_weights)),
            'global_min': float(np.min(all_weights)),
            'global_max': float(np.max(all_weights)),
            'global_abs_max': float(np.max(np.abs(all_weights))),
            'global_zero_fraction': float(np.mean(all_weights == 0.0))
        }
        
        logger.info(f"Weight statistics calculated for {len(weights_dict)} layers")
        logger.info(f"Total parameters: {statistics['overall']['total_parameters']:,}")
        logger.info(f"Total memory: {statistics['overall']['total_memory_mb']:.2f} MB")
        
        return statistics
    
    def flatten_weights_for_storage(
        self, 
        weights_dict: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, Dict[str, Union[int, List[int]]]]]:
        """
        Flatten all weights into a single array for SRAM storage simulation.
        
        Args:
            weights_dict: Dictionary of layer names to weight arrays
            
        Returns:
            Tuple of (flattened_weights, reconstruction_info)
        """
        logger.info("Flattening weights for SRAM storage...")
        
        flattened_weights = []
        reconstruction_info = {}
        current_offset = 0
        
        for layer_name, weights in weights_dict.items():
            flat_weights = weights.flatten()
            flattened_weights.append(flat_weights)
            
            reconstruction_info[layer_name] = {
                'original_shape': list(weights.shape),
                'start_index': current_offset,
                'end_index': current_offset + len(flat_weights),
                'num_elements': len(flat_weights),
                'dtype': str(weights.dtype)
            }
            
            current_offset += len(flat_weights)
        
        # Combine all flattened weights
        all_weights = np.concatenate(flattened_weights)
        
        logger.info(f"Flattened {len(weights_dict)} layers into {len(all_weights):,} elements")
        logger.info(f"Weight range: [{np.min(all_weights):.6f}, {np.max(all_weights):.6f}]")
        
        return all_weights, reconstruction_info
    
    def reconstruct_weights_from_flat(
        self, 
        flat_weights: np.ndarray, 
        reconstruction_info: Dict[str, Dict[str, Union[int, List[int]]]]
    ) -> Dict[str, np.ndarray]:
        """
        Reconstruct original weight structure from flattened array.
        
        Args:
            flat_weights: Flattened weight array
            reconstruction_info: Information needed for reconstruction
            
        Returns:
            Dictionary of reconstructed weights
        """
        logger.info("Reconstructing weights from flattened array...")
        
        reconstructed_weights = {}
        
        for layer_name, info in reconstruction_info.items():
            start_idx = info['start_index']
            end_idx = info['end_index']
            original_shape = info['original_shape']
            
            # Extract and reshape weights
            layer_weights = flat_weights[start_idx:end_idx]
            reconstructed_weights[layer_name] = layer_weights.reshape(original_shape)
        
        logger.info(f"Reconstructed {len(reconstructed_weights)} weight tensors")
        
        return reconstructed_weights
    
    def save_weights_to_files(
        self, 
        weights_dict: Dict[str, np.ndarray],
        save_dir: str,
        include_statistics: bool = True
    ) -> Dict[str, str]:
        """
        Save extracted weights to files for SRAM simulation.
        
        Args:
            weights_dict: Dictionary of weights to save
            save_dir: Directory to save weights
            include_statistics: Whether to save statistics file
            
        Returns:
            Dictionary mapping content type to file paths
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving weights to {save_dir}")
        
        saved_files = {}
        
        # Save individual weight arrays
        weights_file = save_dir / "extracted_weights.npz"
        np.savez_compressed(weights_file, **weights_dict)
        saved_files['weights'] = str(weights_file)
        logger.info(f"Saved weights to {weights_file}")
        
        # Save flattened weights for SRAM simulation
        flat_weights, reconstruction_info = self.flatten_weights_for_storage(weights_dict)
        
        flat_file = save_dir / "flattened_weights.npy"
        np.save(flat_file, flat_weights)
        saved_files['flattened'] = str(flat_file)
        
        reconstruction_file = save_dir / "reconstruction_info.json"
        with open(reconstruction_file, 'w') as f:
            json.dump(reconstruction_info, f, indent=2)
        saved_files['reconstruction_info'] = str(reconstruction_file)
        
        # Save statistics if requested
        if include_statistics:
            statistics = self.get_weight_statistics(weights_dict)
            stats_file = save_dir / "weight_statistics.json"
            with open(stats_file, 'w') as f:
                json.dump(statistics, f, indent=2)
            saved_files['statistics'] = str(stats_file)
            logger.info(f"Saved statistics to {stats_file}")
        
        # Save organized weights by layer type
        organized_weights = self.organize_weights_by_layer_type(weights_dict)
        for layer_type, type_weights in organized_weights.items():
            if type_weights:  # Only save if there are weights of this type
                type_file = save_dir / f"weights_{layer_type}.npz"
                np.savez_compressed(type_file, **type_weights)
                saved_files[f'weights_{layer_type}'] = str(type_file)
        
        logger.info(f"Successfully saved {len(saved_files)} files")
        return saved_files
    
    def load_weights_from_files(self, weights_file: str) -> Dict[str, np.ndarray]:
        """
        Load weights from saved files.
        
        Args:
            weights_file: Path to the weights file (.npz)
            
        Returns:
            Dictionary of loaded weights
        """
        logger.info(f"Loading weights from {weights_file}")
        
        weights_data = np.load(weights_file)
        weights_dict = {key: weights_data[key] for key in weights_data.files}
        
        logger.info(f"Loaded {len(weights_dict)} weight tensors")
        return weights_dict
    
    def compare_weights(
        self, 
        original_weights: Dict[str, np.ndarray],
        modified_weights: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare two sets of weights to quantify differences.
        
        Args:
            original_weights: Original weight dictionary
            modified_weights: Modified weight dictionary (e.g., after SRAM storage)
            
        Returns:
            Dictionary with comparison metrics for each layer
        """
        logger.info("Comparing original and modified weights...")
        
        comparison_results = {}
        
        for layer_name in original_weights.keys():
            if layer_name not in modified_weights:
                logger.warning(f"Layer {layer_name} not found in modified weights")
                continue
            
            orig = original_weights[layer_name]
            mod = modified_weights[layer_name]
            
            if orig.shape != mod.shape:
                logger.warning(f"Shape mismatch for {layer_name}: {orig.shape} vs {mod.shape}")
                continue
            
            # Calculate various difference metrics
            diff = orig - mod
            abs_diff = np.abs(diff)
            rel_diff = np.divide(abs_diff, np.abs(orig) + 1e-8, 
                               out=np.zeros_like(abs_diff), where=np.abs(orig) > 1e-8)
            
            layer_comparison = {
                'mse': float(np.mean(diff ** 2)),
                'mae': float(np.mean(abs_diff)),
                'max_abs_error': float(np.max(abs_diff)),
                'mean_rel_error': float(np.mean(rel_diff)),
                'max_rel_error': float(np.max(rel_diff)),
                'snr_db': float(10 * np.log10(np.mean(orig ** 2) / (np.mean(diff ** 2) + 1e-12))),
                'correlation': float(np.corrcoef(orig.flatten(), mod.flatten())[0, 1]),
                'cosine_similarity': float(np.dot(orig.flatten(), mod.flatten()) / 
                                         (np.linalg.norm(orig.flatten()) * np.linalg.norm(mod.flatten()) + 1e-12))
            }
            
            comparison_results[layer_name] = layer_comparison
        
        # Calculate overall metrics
        all_orig = np.concatenate([w.flatten() for w in original_weights.values()])
        all_mod = np.concatenate([w.flatten() for w in modified_weights.values()])
        
        overall_diff = all_orig - all_mod
        overall_abs_diff = np.abs(overall_diff)
        overall_rel_diff = np.divide(overall_abs_diff, np.abs(all_orig) + 1e-8,
                                   out=np.zeros_like(overall_abs_diff), where=np.abs(all_orig) > 1e-8)
        
        comparison_results['overall'] = {
            'mse': float(np.mean(overall_diff ** 2)),
            'mae': float(np.mean(overall_abs_diff)),
            'max_abs_error': float(np.max(overall_abs_diff)),
            'mean_rel_error': float(np.mean(overall_rel_diff)),
            'max_rel_error': float(np.max(overall_rel_diff)),
            'snr_db': float(10 * np.log10(np.mean(all_orig ** 2) / (np.mean(overall_diff ** 2) + 1e-12))),
            'correlation': float(np.corrcoef(all_orig, all_mod)[0, 1]),
            'cosine_similarity': float(np.dot(all_orig, all_mod) / 
                                     (np.linalg.norm(all_orig) * np.linalg.norm(all_mod) + 1e-12))
        }
        
        logger.info(f"Weight comparison completed for {len(comparison_results)-1} layers")
        logger.info(f"Overall MSE: {comparison_results['overall']['mse']:.2e}")
        logger.info(f"Overall SNR: {comparison_results['overall']['snr_db']:.2f} dB")
        
        return comparison_results


def extract_weights_from_model_file(
    model_file: str,
    model_class: nn.Module,
    save_dir: str,
    **model_kwargs
) -> Dict[str, str]:
    """
    Convenience function to extract weights from a saved model file.
    
    Args:
        model_file: Path to saved model file
        model_class: Model class to instantiate
        save_dir: Directory to save extracted weights
        **model_kwargs: Additional arguments for model instantiation
        
    Returns:
        Dictionary of saved file paths
    """
    logger.info(f"Extracting weights from model file: {model_file}")
    
    # Load model
    checkpoint = torch.load(model_file, map_location='cpu')
    model = model_class(**model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Extract weights
    extractor = WeightExtractor()
    weights_dict = extractor.extract_all_weights(model)
    
    # Save weights
    saved_files = extractor.save_weights_to_files(weights_dict, save_dir)
    
    logger.info(f"Weight extraction completed. Files saved to {save_dir}")
    return saved_files


def create_dummy_sram_weights(
    original_weights: Dict[str, np.ndarray],
    quantization_bits: int = 8,
    noise_std: float = 0.01
) -> Dict[str, np.ndarray]:
    """
    Create dummy SRAM-affected weights for testing (before actual QSPICE simulation).
    
    Args:
        original_weights: Original weight dictionary
        quantization_bits: Number of bits for quantization simulation
        noise_std: Standard deviation of added noise
        
    Returns:
        Dictionary of SRAM-affected weights
    """
    logger.info(f"Creating dummy SRAM weights with {quantization_bits}-bit quantization")
    
    sram_weights = {}
    
    for layer_name, weights in original_weights.items():
        # Simulate quantization
        weight_min, weight_max = np.min(weights), np.max(weights)
        weight_range = weight_max - weight_min
        
        # Quantize to specified bits
        quantized = np.round((weights - weight_min) / weight_range * (2**quantization_bits - 1))
        dequantized = quantized / (2**quantization_bits - 1) * weight_range + weight_min
        
        # Add noise to simulate SRAM effects
        noise = np.random.normal(0, noise_std, weights.shape)
        sram_affected = dequantized + noise
        
        sram_weights[layer_name] = sram_affected.astype(weights.dtype)
    
    logger.info("Dummy SRAM weights created")
    return sram_weights