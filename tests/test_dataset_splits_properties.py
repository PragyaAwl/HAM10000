"""
Property-based tests for dataset splitting.

Tests Property 3: Stratified split preservation
Validates: Requirements 1.3
"""

import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from collections import Counter
from typing import List, Dict

from src.dataset_splits import HAM10000DatasetSplitter, create_ham10000_splits


class TestDatasetSplitsProperties:
    """Property-based tests for dataset splitting."""
    
    @given(
        n_samples=st.integers(min_value=300, max_value=1000),
        test_size=st.floats(min_value=0.15, max_value=0.25),
        val_size=st.floats(min_value=0.0, max_value=0.1),
        n_classes=st.integers(min_value=3, max_value=5)
    )
    @settings(max_examples=50, deadline=None)
    def test_property_3_stratified_split_preservation(self, n_samples, test_size, val_size, n_classes):
        """
        Property 3: Stratified split preservation
        
        For any dataset split operation, the class distribution in training, 
        validation, and test sets should match the original dataset distribution 
        within acceptable tolerance.
        
        Validates: Requirements 1.3
        """
        # Ensure we have enough samples for stratified splitting
        assume(test_size + val_size < 0.4)  # Leave enough for training
        
        # Calculate minimum samples needed per class for stratified splitting
        min_test_samples = max(2, int(n_samples * test_size / n_classes))
        min_val_samples = max(2, int(n_samples * val_size / n_classes)) if val_size > 0.02 else 0
        min_samples_per_class = max(15, min_test_samples + min_val_samples + 8)
        
        assume(n_samples >= n_classes * min_samples_per_class)
        
        # Skip validation if it would be too small
        if val_size > 0 and n_samples * val_size < n_classes * 3:
            val_size = 0.0
        
        # Create balanced class distribution with some variation
        base_size = n_samples // n_classes
        class_sizes = []
        remaining_samples = n_samples
        
        for i in range(n_classes - 1):
            # Add some variation but keep reasonable balance
            variation = int(base_size * 0.2)  # 20% variation
            size = base_size + np.random.randint(-variation, variation + 1)
            size = max(min_samples_per_class, min(size, remaining_samples - (n_classes - i - 1) * min_samples_per_class))
            class_sizes.append(size)
            remaining_samples -= size
        
        # Last class gets remaining samples
        class_sizes.append(remaining_samples)
        
        # Create class labels
        class_names = ['mel', 'nv', 'bcc', 'akiec', 'bkl'][:n_classes]
        labels = []
        for class_name, size in zip(class_names, class_sizes):
            labels.extend([class_name] * size)
        
        # Shuffle labels
        np.random.shuffle(labels)
        
        # Create dummy metadata
        metadata = pd.DataFrame({
            'lesion_id': [f"HAM_{i:07d}" for i in range(len(labels))],
            'image_id': [f"ISIC_{i:07d}" for i in range(len(labels))],
            'dx': labels,
            'dx_type': ['histo'] * len(labels),
            'age': np.random.normal(50, 15, len(labels)),
            'sex': np.random.choice(['male', 'female'], len(labels)),
            'localization': np.random.choice(['scalp', 'face', 'back'], len(labels))
        })
        
        # Create splitter
        splitter = HAM10000DatasetSplitter(random_state=42)
        
        # Create splits
        train_metadata, val_metadata, test_metadata = splitter.create_stratified_split(
            metadata, test_size=test_size, val_size=val_size
        )
        
        # Property 1: No overlapping samples between splits
        train_ids = set(train_metadata['image_id'])
        val_ids = set(val_metadata['image_id']) if len(val_metadata) > 0 else set()
        test_ids = set(test_metadata['image_id'])
        
        assert len(train_ids.intersection(val_ids)) == 0, "Train and validation sets should not overlap"
        assert len(train_ids.intersection(test_ids)) == 0, "Train and test sets should not overlap"
        assert len(val_ids.intersection(test_ids)) == 0, "Validation and test sets should not overlap"
        
        # Property 2: All samples should be accounted for
        total_split_samples = len(train_metadata) + len(val_metadata) + len(test_metadata)
        assert total_split_samples == len(metadata), "All samples should be accounted for in splits"
        
        # Property 3: Class distribution should be preserved within reasonable tolerance
        original_dist = Counter(metadata['dx'])
        train_dist = Counter(train_metadata['dx'])
        test_dist = Counter(test_metadata['dx'])
        
        # Calculate proportions with more lenient tolerance for smaller datasets
        total_samples = len(metadata)
        tolerance = max(0.15, 3.0 / min(class_sizes))  # Adaptive tolerance based on class size
        
        for class_name in original_dist.keys():
            original_prop = original_dist[class_name] / total_samples
            
            # Check train distribution
            train_prop = train_dist.get(class_name, 0) / len(train_metadata)
            train_diff = abs(train_prop - original_prop)
            assert train_diff <= tolerance, f"Train class {class_name} distribution should be preserved within {tolerance:.3f}: {train_diff:.3f}"
            
            # Check test distribution
            test_prop = test_dist.get(class_name, 0) / len(test_metadata)
            test_diff = abs(test_prop - original_prop)
            assert test_diff <= tolerance, f"Test class {class_name} distribution should be preserved within {tolerance:.3f}: {test_diff:.3f}"
        
        # Property 4: All classes should be represented in major splits
        train_classes = set(train_metadata['dx'])
        test_classes = set(test_metadata['dx'])
        original_classes = set(metadata['dx'])
        
        assert train_classes == original_classes, "All classes should be represented in training set"
        assert test_classes == original_classes, "All classes should be represented in test set"
        
        # Property 5: Split sizes should be approximately correct
        expected_test_size = int(len(metadata) * test_size)
        expected_val_size = int(len(metadata) * val_size) if val_size > 0 else 0
        
        # Allow reasonable tolerance for rounding and stratification constraints
        size_tolerance = max(n_classes, int(len(metadata) * 0.05))  # 5% tolerance or n_classes, whichever is larger
        
        test_size_diff = abs(len(test_metadata) - expected_test_size)
        assert test_size_diff <= size_tolerance, f"Test set size should match expected size within tolerance: expected={expected_test_size}, actual={len(test_metadata)}, diff={test_size_diff}, tolerance={size_tolerance}"
    
    @given(
        n_samples=st.integers(min_value=50, max_value=500),
        missing_age_rate=st.floats(min_value=0.0, max_value=0.3),
        missing_sex_rate=st.floats(min_value=0.0, max_value=0.1),
        exclude_missing_age=st.booleans(),
        exclude_missing_sex=st.booleans()
    )
    @settings(max_examples=50)
    def test_property_3_missing_data_handling(self, n_samples, missing_age_rate, missing_sex_rate, exclude_missing_age, exclude_missing_sex):
        """
        Property 3 extension: Missing data handling consistency
        
        For any dataset with missing values, cleaning should either impute or 
        exclude incomplete records while maintaining the total valid record count.
        
        Validates: Requirements 1.4
        """
        # Create dummy metadata with missing values
        labels = np.random.choice(['mel', 'nv', 'bcc', 'akiec', 'bkl'], n_samples)
        
        metadata = pd.DataFrame({
            'lesion_id': [f"HAM_{i:07d}" for i in range(n_samples)],
            'image_id': [f"ISIC_{i:07d}" for i in range(n_samples)],
            'dx': labels,
            'dx_type': ['histo'] * n_samples,
            'age': np.random.normal(50, 15, n_samples),
            'sex': np.random.choice(['male', 'female'], n_samples),
            'localization': np.random.choice(['scalp', 'face'], n_samples)
        })
        
        # Introduce missing values
        n_missing_age = int(n_samples * missing_age_rate)
        n_missing_sex = int(n_samples * missing_sex_rate)
        
        if n_missing_age > 0:
            missing_age_indices = np.random.choice(n_samples, n_missing_age, replace=False)
            metadata.loc[missing_age_indices, 'age'] = np.nan
        
        if n_missing_sex > 0:
            missing_sex_indices = np.random.choice(n_samples, n_missing_sex, replace=False)
            metadata.loc[missing_sex_indices, 'sex'] = np.nan
        
        # Create splitter and clean data
        splitter = HAM10000DatasetSplitter(random_state=42)
        
        cleaned_metadata = splitter.clean_metadata(
            metadata,
            exclude_missing_age=exclude_missing_age,
            exclude_missing_sex=exclude_missing_sex
        )
        
        # Property 1: Cleaned data should have no missing values in excluded columns
        if exclude_missing_age:
            assert cleaned_metadata['age'].isna().sum() == 0, "Cleaned data should have no missing age values when excluded"
        
        if exclude_missing_sex:
            assert cleaned_metadata['sex'].isna().sum() == 0, "Cleaned data should have no missing sex values when excluded"
        
        # Property 2: Required columns should never have missing values
        required_columns = ['lesion_id', 'image_id', 'dx']
        for col in required_columns:
            assert cleaned_metadata[col].isna().sum() == 0, f"Required column {col} should have no missing values"
        
        # Property 3: No duplicate image_ids should remain
        assert cleaned_metadata['image_id'].duplicated().sum() == 0, "No duplicate image_ids should remain after cleaning"
        
        # Property 4: Cleaned data should be splittable if it has enough samples
        if len(cleaned_metadata) >= 20:  # Minimum for meaningful split
            try:
                train_meta, val_meta, test_meta = splitter.create_stratified_split(
                    cleaned_metadata, test_size=0.2, val_size=0.1
                )
                
                # All splits should contain only clean data
                for split_meta in [train_meta, val_meta, test_meta]:
                    if len(split_meta) > 0:
                        if exclude_missing_age:
                            assert split_meta['age'].isna().sum() == 0, "Split should have no missing age"
                        if exclude_missing_sex:
                            assert split_meta['sex'].isna().sum() == 0, "Split should have no missing sex"
                        
                        for col in required_columns:
                            assert split_meta[col].isna().sum() == 0, f"Split should have no missing {col}"
                            
            except ValueError as e:
                # This is acceptable if there are insufficient samples for stratification
                assert "Insufficient samples" in str(e), f"Unexpected error during splitting: {e}"
    
    @given(
        class_imbalance_ratio=st.floats(min_value=1.5, max_value=10.0),
        n_classes=st.integers(min_value=3, max_value=7),
        total_samples=st.integers(min_value=200, max_value=800)
    )
    @settings(max_examples=30)
    def test_property_3_class_imbalance_handling(self, class_imbalance_ratio, n_classes, total_samples):
        """
        Property 3 extension: Class imbalance handling
        
        For any dataset with class imbalance, stratified splitting should 
        preserve the imbalance ratio across all splits.
        
        Validates: Requirements 1.3
        """
        # Create imbalanced class distribution
        class_names = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc'][:n_classes]
        
        # Create exponentially decreasing class sizes
        base_size = total_samples // (n_classes * 2)
        class_sizes = []
        remaining_samples = total_samples
        
        for i in range(n_classes - 1):
            if i == 0:
                # Largest class
                size = int(base_size * class_imbalance_ratio)
            else:
                # Decreasing sizes
                size = max(10, int(base_size * (class_imbalance_ratio ** (1 - i/n_classes))))
            
            size = min(size, remaining_samples - (n_classes - i - 1) * 10)  # Ensure minimum for remaining classes
            class_sizes.append(size)
            remaining_samples -= size
        
        # Last class gets remaining samples
        class_sizes.append(max(10, remaining_samples))
        
        # Create labels
        labels = []
        for class_name, size in zip(class_names, class_sizes):
            labels.extend([class_name] * size)
        
        np.random.shuffle(labels)
        
        # Create metadata
        metadata = pd.DataFrame({
            'lesion_id': [f"HAM_{i:07d}" for i in range(len(labels))],
            'image_id': [f"ISIC_{i:07d}" for i in range(len(labels))],
            'dx': labels,
            'dx_type': ['histo'] * len(labels),
            'age': np.random.normal(50, 15, len(labels)),
            'sex': np.random.choice(['male', 'female'], len(labels)),
            'localization': np.random.choice(['scalp', 'face'], len(labels))
        })
        
        # Calculate original imbalance ratio
        original_counts = Counter(metadata['dx'])
        original_max = max(original_counts.values())
        original_min = min(original_counts.values())
        original_imbalance = original_max / original_min
        
        # Create splits
        splitter = HAM10000DatasetSplitter(random_state=42)
        train_metadata, val_metadata, test_metadata = splitter.create_stratified_split(
            metadata, test_size=0.2, val_size=0.1
        )
        
        # Property: Imbalance ratio should be preserved in each split
        tolerance = 0.5  # Allow some tolerance for small sample effects
        
        for split_name, split_data in [('train', train_metadata), ('test', test_metadata)]:
            if len(split_data) > 0:
                split_counts = Counter(split_data['dx'])
                split_max = max(split_counts.values())
                split_min = min(split_counts.values())
                split_imbalance = split_max / split_min
                
                imbalance_diff = abs(split_imbalance - original_imbalance)
                relative_diff = imbalance_diff / original_imbalance
                
                assert relative_diff <= tolerance, f"{split_name} split imbalance ratio should be preserved within {tolerance*100}%: original={original_imbalance:.2f}, split={split_imbalance:.2f}"
        
        # Property: All classes should be represented in major splits
        train_classes = set(train_metadata['dx'])
        test_classes = set(test_metadata['dx'])
        original_classes = set(metadata['dx'])
        
        assert train_classes == original_classes, "All classes should be represented in training set"
        assert test_classes == original_classes, "All classes should be represented in test set"


if __name__ == "__main__":
    # Run property tests
    pytest.main([__file__, "-v"])