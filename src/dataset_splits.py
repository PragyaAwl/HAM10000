"""
Dataset Splits and Validation for HAM10000

This module implements stratified train/test split maintaining class balance,
handles missing data, and validates data integrity.

Requirements: 1.3, 1.4
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from typing import Dict, List, Tuple, Optional, Set
import logging
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HAM10000DatasetSplitter:
    """
    Dataset splitter for HAM10000 with stratified sampling and missing data handling.
    
    Handles creating train/validation/test splits while maintaining class balance
    and excluding incomplete records.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the dataset splitter.
        
        Args:
            random_state: Random seed for reproducible splits
        """
        self.random_state = random_state
        self.split_info = {}
        
    def clean_metadata(self, metadata: pd.DataFrame, 
                      required_columns: List[str] = None,
                      exclude_missing_age: bool = True,
                      exclude_missing_sex: bool = True,
                      exclude_missing_localization: bool = False) -> pd.DataFrame:
        """
        Clean metadata by handling missing data.
        
        Args:
            metadata: Input metadata DataFrame
            required_columns: List of columns that must not be missing
            exclude_missing_age: Whether to exclude records with missing age
            exclude_missing_sex: Whether to exclude records with missing sex
            exclude_missing_localization: Whether to exclude records with missing localization
            
        Returns:
            Cleaned metadata DataFrame
        """
        if required_columns is None:
            required_columns = ['lesion_id', 'image_id', 'dx']
        
        original_count = len(metadata)
        cleaned_metadata = metadata.copy()
        
        logger.info(f"Starting with {original_count} records")
        
        # Check for missing values in required columns
        for col in required_columns:
            if col not in cleaned_metadata.columns:
                raise ValueError(f"Required column '{col}' not found in metadata")
            
            missing_count = cleaned_metadata[col].isna().sum()
            if missing_count > 0:
                logger.warning(f"Found {missing_count} missing values in required column '{col}'")
                cleaned_metadata = cleaned_metadata.dropna(subset=[col])
                logger.info(f"After removing missing {col}: {len(cleaned_metadata)} records")
        
        # Handle optional missing data exclusions
        if exclude_missing_age and 'age' in cleaned_metadata.columns:
            missing_age = cleaned_metadata['age'].isna().sum()
            if missing_age > 0:
                logger.info(f"Excluding {missing_age} records with missing age")
                cleaned_metadata = cleaned_metadata.dropna(subset=['age'])
                logger.info(f"After removing missing age: {len(cleaned_metadata)} records")
        
        if exclude_missing_sex and 'sex' in cleaned_metadata.columns:
            missing_sex = cleaned_metadata['sex'].isna().sum()
            if missing_sex > 0:
                logger.info(f"Excluding {missing_sex} records with missing sex")
                cleaned_metadata = cleaned_metadata.dropna(subset=['sex'])
                logger.info(f"After removing missing sex: {len(cleaned_metadata)} records")
        
        if exclude_missing_localization and 'localization' in cleaned_metadata.columns:
            missing_loc = cleaned_metadata['localization'].isna().sum()
            if missing_loc > 0:
                logger.info(f"Excluding {missing_loc} records with missing localization")
                cleaned_metadata = cleaned_metadata.dropna(subset=['localization'])
                logger.info(f"After removing missing localization: {len(cleaned_metadata)} records")
        
        # Remove duplicates based on image_id (keep first occurrence)
        duplicate_count = cleaned_metadata.duplicated(subset=['image_id']).sum()
        if duplicate_count > 0:
            logger.info(f"Removing {duplicate_count} duplicate image_id records")
            cleaned_metadata = cleaned_metadata.drop_duplicates(subset=['image_id'], keep='first')
            logger.info(f"After removing duplicates: {len(cleaned_metadata)} records")
        
        # Reset index
        cleaned_metadata = cleaned_metadata.reset_index(drop=True)
        
        removed_count = original_count - len(cleaned_metadata)
        logger.info(f"Data cleaning complete: removed {removed_count} records ({removed_count/original_count*100:.1f}%)")
        
        return cleaned_metadata
    
    def validate_class_distribution(self, labels: List[str], 
                                  min_samples_per_class: int = 2) -> Dict[str, int]:
        """
        Validate class distribution and identify classes with insufficient samples.
        
        Args:
            labels: List of class labels
            min_samples_per_class: Minimum samples required per class
            
        Returns:
            Dictionary with class counts
            
        Raises:
            ValueError: If any class has insufficient samples
        """
        class_counts = Counter(labels)
        
        logger.info("Class distribution:")
        for class_name, count in sorted(class_counts.items()):
            logger.info(f"  {class_name}: {count} samples")
        
        # Check for classes with insufficient samples
        insufficient_classes = {cls: count for cls, count in class_counts.items() 
                              if count < min_samples_per_class}
        
        if insufficient_classes:
            logger.error(f"Classes with insufficient samples (< {min_samples_per_class}): {insufficient_classes}")
            raise ValueError(f"Insufficient samples for stratified split: {insufficient_classes}")
        
        return dict(class_counts)
    
    def create_stratified_split(self, metadata: pd.DataFrame,
                              test_size: float = 0.2,
                              val_size: float = 0.1,
                              stratify_column: str = 'dx') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create stratified train/validation/test splits.
        
        Args:
            metadata: Cleaned metadata DataFrame
            test_size: Proportion of data for test set
            val_size: Proportion of data for validation set (from remaining after test)
            stratify_column: Column to use for stratification
            
        Returns:
            Tuple of (train_metadata, val_metadata, test_metadata)
        """
        if stratify_column not in metadata.columns:
            raise ValueError(f"Stratification column '{stratify_column}' not found in metadata")
        
        # Validate class distribution
        labels = metadata[stratify_column].tolist()
        class_counts = self.validate_class_distribution(labels)
        
        # First split: separate test set
        train_val_metadata, test_metadata = train_test_split(
            metadata,
            test_size=test_size,
            stratify=metadata[stratify_column],
            random_state=self.random_state
        )
        
        # Second split: separate validation from training
        if val_size > 0:
            # Adjust validation size relative to remaining data
            val_size_adjusted = val_size / (1 - test_size)
            
            # Ensure minimum samples per class for validation split
            min_val_samples_per_class = 2
            min_val_total = min_val_samples_per_class * len(class_counts)
            
            if len(train_val_metadata) * val_size_adjusted < min_val_total:
                # Skip validation split if too small
                logger.warning(f"Validation split too small ({len(train_val_metadata) * val_size_adjusted:.1f} samples), skipping validation set")
                train_metadata = train_val_metadata
                val_metadata = pd.DataFrame(columns=metadata.columns)
            else:
                train_metadata, val_metadata = train_test_split(
                    train_val_metadata,
                    test_size=val_size_adjusted,
                    stratify=train_val_metadata[stratify_column],
                    random_state=self.random_state
                )
        else:
            train_metadata = train_val_metadata
            val_metadata = pd.DataFrame(columns=metadata.columns)
        
        # Store split information
        self.split_info = {
            'total_samples': len(metadata),
            'train_samples': len(train_metadata),
            'val_samples': len(val_metadata),
            'test_samples': len(test_metadata),
            'test_size': test_size,
            'val_size': val_size,
            'stratify_column': stratify_column,
            'class_counts_total': class_counts,
            'class_counts_train': dict(Counter(train_metadata[stratify_column])),
            'class_counts_val': dict(Counter(val_metadata[stratify_column])) if len(val_metadata) > 0 else {},
            'class_counts_test': dict(Counter(test_metadata[stratify_column]))
        }
        
        logger.info(f"Dataset split complete:")
        logger.info(f"  Total: {len(metadata)} samples")
        logger.info(f"  Train: {len(train_metadata)} samples ({len(train_metadata)/len(metadata)*100:.1f}%)")
        logger.info(f"  Val: {len(val_metadata)} samples ({len(val_metadata)/len(metadata)*100:.1f}%)")
        logger.info(f"  Test: {len(test_metadata)} samples ({len(test_metadata)/len(metadata)*100:.1f}%)")
        
        return train_metadata, val_metadata, test_metadata
    
    def validate_split_integrity(self, train_metadata: pd.DataFrame,
                                val_metadata: pd.DataFrame,
                                test_metadata: pd.DataFrame,
                                tolerance: float = 0.15) -> Dict[str, bool]:
        """
        Validate the integrity of dataset splits.
        
        Args:
            train_metadata: Training set metadata
            val_metadata: Validation set metadata  
            test_metadata: Test set metadata
            tolerance: Tolerance for class distribution differences
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {}
        
        # Check for overlapping samples
        train_ids = set(train_metadata['image_id'])
        val_ids = set(val_metadata['image_id']) if len(val_metadata) > 0 else set()
        test_ids = set(test_metadata['image_id'])
        
        train_val_overlap = train_ids.intersection(val_ids)
        train_test_overlap = train_ids.intersection(test_ids)
        val_test_overlap = val_ids.intersection(test_ids)
        
        validation_results['no_train_val_overlap'] = len(train_val_overlap) == 0
        validation_results['no_train_test_overlap'] = len(train_test_overlap) == 0
        validation_results['no_val_test_overlap'] = len(val_test_overlap) == 0
        
        if train_val_overlap:
            logger.error(f"Found {len(train_val_overlap)} overlapping samples between train and val")
        if train_test_overlap:
            logger.error(f"Found {len(train_test_overlap)} overlapping samples between train and test")
        if val_test_overlap:
            logger.error(f"Found {len(val_test_overlap)} overlapping samples between val and test")
        
        # Check class distribution preservation
        if 'class_counts_total' in self.split_info:
            total_dist = self.split_info['class_counts_total']
            train_dist = self.split_info['class_counts_train']
            test_dist = self.split_info['class_counts_test']
            
            # Calculate proportions
            total_samples = sum(total_dist.values())
            train_samples = sum(train_dist.values())
            test_samples = sum(test_dist.values())
            
            class_distribution_preserved = True
            
            for class_name in total_dist.keys():
                total_prop = total_dist[class_name] / total_samples
                train_prop = train_dist.get(class_name, 0) / train_samples
                test_prop = test_dist.get(class_name, 0) / test_samples
                
                # Check if proportions are within tolerance
                train_diff = abs(train_prop - total_prop)
                test_diff = abs(test_prop - total_prop)
                
                if train_diff > tolerance or test_diff > tolerance:
                    class_distribution_preserved = False
                    logger.warning(f"Class {class_name} distribution not preserved within tolerance {tolerance}")
                    logger.warning(f"  Total: {total_prop:.3f}, Train: {train_prop:.3f}, Test: {test_prop:.3f}")
            
            validation_results['class_distribution_preserved'] = class_distribution_preserved
        
        # Overall validation
        validation_results['valid'] = all([
            validation_results['no_train_val_overlap'],
            validation_results['no_train_test_overlap'],
            validation_results['no_val_test_overlap'],
            validation_results.get('class_distribution_preserved', True)
        ])
        
        return validation_results
    
    def get_split_statistics(self) -> Dict:
        """
        Get comprehensive statistics about the dataset split.
        
        Returns:
            Dictionary with split statistics
        """
        if not self.split_info:
            return {}
        
        stats = self.split_info.copy()
        
        # Add proportion information
        total = stats['total_samples']
        stats['train_proportion'] = stats['train_samples'] / total
        stats['val_proportion'] = stats['val_samples'] / total
        stats['test_proportion'] = stats['test_samples'] / total
        
        # Add class balance information
        if 'class_counts_total' in stats:
            total_classes = len(stats['class_counts_total'])
            stats['num_classes'] = total_classes
            
            # Calculate class balance metrics
            class_counts = list(stats['class_counts_total'].values())
            stats['class_balance'] = {
                'min_samples': min(class_counts),
                'max_samples': max(class_counts),
                'mean_samples': np.mean(class_counts),
                'std_samples': np.std(class_counts),
                'imbalance_ratio': max(class_counts) / min(class_counts)
            }
        
        return stats
    
    def save_splits(self, train_metadata: pd.DataFrame,
                   val_metadata: pd.DataFrame,
                   test_metadata: pd.DataFrame,
                   output_dir: str = "data/splits") -> Dict[str, str]:
        """
        Save dataset splits to CSV files.
        
        Args:
            train_metadata: Training set metadata
            val_metadata: Validation set metadata
            test_metadata: Test set metadata
            output_dir: Directory to save split files
            
        Returns:
            Dictionary with file paths
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        file_paths = {}
        
        # Save train split
        train_path = os.path.join(output_dir, "train_metadata.csv")
        train_metadata.to_csv(train_path, index=False)
        file_paths['train'] = train_path
        
        # Save validation split (if not empty)
        if len(val_metadata) > 0:
            val_path = os.path.join(output_dir, "val_metadata.csv")
            val_metadata.to_csv(val_path, index=False)
            file_paths['val'] = val_path
        
        # Save test split
        test_path = os.path.join(output_dir, "test_metadata.csv")
        test_metadata.to_csv(test_path, index=False)
        file_paths['test'] = test_path
        
        # Save split statistics
        stats_path = os.path.join(output_dir, "split_statistics.json")
        import json
        with open(stats_path, 'w') as f:
            json.dump(self.get_split_statistics(), f, indent=2)
        file_paths['statistics'] = stats_path
        
        logger.info(f"Saved dataset splits to {output_dir}")
        
        return file_paths


def create_ham10000_splits(metadata: pd.DataFrame,
                          test_size: float = 0.2,
                          val_size: float = 0.1,
                          random_state: int = 42,
                          clean_data: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to create HAM10000 dataset splits.
    
    Args:
        metadata: HAM10000 metadata DataFrame
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed
        clean_data: Whether to clean data before splitting
        
    Returns:
        Tuple of (train_metadata, val_metadata, test_metadata)
    """
    splitter = HAM10000DatasetSplitter(random_state=random_state)
    
    # Clean data if requested
    if clean_data:
        cleaned_metadata = splitter.clean_metadata(metadata)
    else:
        cleaned_metadata = metadata
    
    # Create splits
    train_metadata, val_metadata, test_metadata = splitter.create_stratified_split(
        cleaned_metadata, test_size=test_size, val_size=val_size
    )
    
    # Validate splits
    validation = splitter.validate_split_integrity(train_metadata, val_metadata, test_metadata)
    
    if not validation['valid']:
        logger.warning("Split validation failed - check logs for details")
    else:
        logger.info("Split validation passed")
    
    return train_metadata, val_metadata, test_metadata


if __name__ == "__main__":
    # Example usage with dummy data
    np.random.seed(42)
    
    # Create dummy metadata
    n_samples = 1000
    dummy_metadata = pd.DataFrame({
        'lesion_id': [f"HAM_{i:07d}" for i in range(n_samples)],
        'image_id': [f"ISIC_{i:07d}" for i in range(n_samples)],
        'dx': np.random.choice(['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc'], n_samples),
        'dx_type': np.random.choice(['histo', 'follow_up', 'consensus'], n_samples),
        'age': np.random.normal(50, 15, n_samples),
        'sex': np.random.choice(['male', 'female'], n_samples),
        'localization': np.random.choice(['scalp', 'face', 'back', 'trunk'], n_samples)
    })
    
    # Add some missing values
    dummy_metadata.loc[np.random.choice(n_samples, 50, replace=False), 'age'] = np.nan
    
    # Create splits
    train_meta, val_meta, test_meta = create_ham10000_splits(dummy_metadata)
    
    print(f"Split sizes - Train: {len(train_meta)}, Val: {len(val_meta)}, Test: {len(test_meta)}")
    
    # Show class distributions
    print("\nClass distributions:")
    print("Train:", dict(Counter(train_meta['dx'])))
    print("Val:", dict(Counter(val_meta['dx'])))
    print("Test:", dict(Counter(test_meta['dx'])))