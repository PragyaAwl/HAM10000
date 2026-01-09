"""
Property-based tests for HAM10000 data loader.

Tests Property 1: Metadata parsing completeness
Validates: Requirements 1.1
"""

import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from pathlib import Path
import tempfile
import os
from typing import List, Dict

from src.data_loader import HAM10000DataLoader, create_ham10000_loader


class TestHAM10000DataLoaderProperties:
    """Property-based tests for HAM10000DataLoader."""
    
    @given(
        diagnoses=st.lists(st.sampled_from(['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']), min_size=1, max_size=100),
        dx_types=st.lists(st.sampled_from(['histo', 'follow_up', 'consensus']), min_size=1, max_size=100),
        ages=st.lists(st.one_of(st.floats(min_value=0, max_value=120), st.none()), min_size=1, max_size=100),
        sexes=st.lists(st.sampled_from(['male', 'female']), min_size=1, max_size=100),
        localizations=st.lists(st.sampled_from(['scalp', 'face', 'back', 'trunk', 'chest', 'upper extremity', 'lower extremity']), min_size=1, max_size=100)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_1_metadata_parsing_completeness(self, diagnoses, dx_types, ages, sexes, localizations):
        """
        Property 1: Metadata parsing completeness
        
        For any valid HAM10000 metadata file, parsing should extract all lesion 
        classifications and patient information without data loss.
        
        Validates: Requirements 1.1
        """
        # Ensure all lists have the same length
        min_length = min(len(diagnoses), len(dx_types), len(ages), len(sexes), len(localizations))
        assume(min_length > 0)
        
        # Truncate all lists to the same length
        diagnoses = diagnoses[:min_length]
        dx_types = dx_types[:min_length]
        ages = ages[:min_length]
        sexes = sexes[:min_length]
        localizations = localizations[:min_length]
        
        # Generate realistic lesion and image IDs
        lesion_ids = [f"HAM_{i:07d}" for i in range(min_length)]
        image_ids = [f"ISIC_{i:07d}" for i in range(min_length)]
        
        # Create temporary metadata CSV
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            
            # Create metadata DataFrame
            metadata_df = pd.DataFrame({
                'lesion_id': lesion_ids,
                'image_id': image_ids,
                'dx': diagnoses,
                'dx_type': dx_types,
                'age': ages,
                'sex': sexes,
                'localization': localizations
            })
            
            # Save to CSV
            metadata_path = temp_dir / "test_metadata.csv"
            metadata_df.to_csv(metadata_path, index=False)
            
            # Create dummy image directories
            images_part1_path = temp_dir / "images_part1"
            images_part2_path = temp_dir / "images_part2"
            images_part1_path.mkdir()
            images_part2_path.mkdir()
            
            # Create loader
            loader = HAM10000DataLoader(
                metadata_path=str(metadata_path),
                images_part1_path=str(images_part1_path),
                images_part2_path=str(images_part2_path)
            )
            
            # Load metadata
            loaded_metadata = loader.load_metadata()
            
            # Property assertions: All data should be preserved
            assert len(loaded_metadata) == len(metadata_df), "Row count should be preserved"
            assert list(loaded_metadata.columns) == list(metadata_df.columns), "Column structure should be preserved"
            
            # Core requirement: All lesion classifications should be preserved
            original_dx_counts = metadata_df['dx'].value_counts().to_dict()
            loaded_dx_counts = loaded_metadata['dx'].value_counts().to_dict()
            assert original_dx_counts == loaded_dx_counts, "Lesion classification distribution should be preserved"
            
            # Core requirement: All patient information should be preserved
            # Check that the same number of records exist for each category
            original_sex_counts = metadata_df['sex'].value_counts().to_dict()
            loaded_sex_counts = loaded_metadata['sex'].value_counts().to_dict()
            assert original_sex_counts == loaded_sex_counts, "Sex distribution should be preserved"
            
            original_loc_counts = metadata_df['localization'].value_counts().to_dict()
            loaded_loc_counts = loaded_metadata['localization'].value_counts().to_dict()
            assert original_loc_counts == loaded_loc_counts, "Localization distribution should be preserved"
            
            # Age statistics should be preserved (handling NaN values)
            original_age_mean = metadata_df['age'].mean()
            loaded_age_mean = loaded_metadata['age'].mean()
            if pd.isna(original_age_mean) and pd.isna(loaded_age_mean):
                pass  # Both NaN is fine
            elif pd.isna(original_age_mean) or pd.isna(loaded_age_mean):
                assert False, "Age mean NaN status should be preserved"
            else:
                assert abs(original_age_mean - loaded_age_mean) < 1e-10, "Age mean should be preserved"
            
            # Check that NaN counts are preserved
            original_age_nan_count = metadata_df['age'].isna().sum()
            loaded_age_nan_count = loaded_metadata['age'].isna().sum()
            assert original_age_nan_count == loaded_age_nan_count, "Age NaN count should be preserved"

    @given(
        diagnoses=st.lists(st.sampled_from(['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']), min_size=1, max_size=100)
    )
    @settings(max_examples=100)
    def test_property_1_label_encoding_consistency(self, diagnoses):
        """
        Property 1 extension: Label encoding consistency
        
        For any set of diagnosis codes, encoding should be consistent and reversible.
        
        Validates: Requirements 1.5
        """
        # Create a minimal loader for testing (using dummy paths)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            
            # Create minimal metadata file
            metadata_df = pd.DataFrame({
                'lesion_id': ['test'] * len(diagnoses),
                'image_id': ['test'] * len(diagnoses),
                'dx': diagnoses,
                'dx_type': ['histo'] * len(diagnoses),
                'age': [50.0] * len(diagnoses),
                'sex': ['male'] * len(diagnoses),
                'localization': ['back'] * len(diagnoses)
            })
            
            metadata_path = temp_dir / "test_metadata.csv"
            metadata_df.to_csv(metadata_path, index=False)
            
            images_part1_path = temp_dir / "images_part1"
            images_part2_path = temp_dir / "images_part2"
            images_part1_path.mkdir()
            images_part2_path.mkdir()
            
            loader = HAM10000DataLoader(
                metadata_path=str(metadata_path),
                images_part1_path=str(images_part1_path),
                images_part2_path=str(images_part2_path)
            )
            
            # Test encoding consistency
            encoded1 = loader.encode_labels(diagnoses)
            encoded2 = loader.encode_labels(diagnoses)
            
            # Property: Encoding should be consistent
            np.testing.assert_array_equal(encoded1, encoded2, "Encoding should be consistent across calls")
            
            # Property: Encoding should be reversible for valid labels
            decoded = loader.decode_labels(encoded1)
            assert len(decoded) == len(diagnoses), "Decoded length should match original"
            
            for orig, dec in zip(diagnoses, decoded):
                assert orig == dec, f"Decoding should be reversible: {orig} != {dec}"
            
            # Property: All encoded values should be valid indices
            valid_indices = set(range(len(HAM10000DataLoader.LESION_CLASSES)))
            encoded_set = set(encoded1[encoded1 >= 0])  # Exclude -1 for unknown labels
            assert encoded_set.issubset(valid_indices), "All encoded values should be valid class indices"

    @given(
        sample_size=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=20)
    def test_property_1_real_dataset_completeness(self, sample_size):
        """
        Property 1 validation: Real dataset parsing completeness
        
        For the actual HAM10000 dataset, parsing should extract all information completely.
        
        Validates: Requirements 1.1
        """
        try:
            # Try to load the real dataset
            loader = create_ham10000_loader()
            metadata = loader.load_metadata()
            
            # Property: All required columns should be present
            required_columns = ['lesion_id', 'image_id', 'dx', 'dx_type', 'age', 'sex', 'localization']
            for col in required_columns:
                assert col in metadata.columns, f"Required column {col} should be present"
            
            # Property: All diagnosis classes should be valid
            unique_dx = set(metadata['dx'].unique())
            expected_dx = set(HAM10000DataLoader.LESION_CLASSES)
            assert unique_dx.issubset(expected_dx), f"All diagnosis classes should be valid: {unique_dx - expected_dx}"
            
            # Property: No completely empty rows
            assert not metadata.isnull().all(axis=1).any(), "No completely empty rows should exist"
            
            # Property: Essential columns should not be completely empty
            essential_columns = ['lesion_id', 'image_id', 'dx']
            for col in essential_columns:
                assert not metadata[col].isnull().all(), f"Essential column {col} should not be completely empty"
            
            # Test with a sample of the data
            sample_metadata = metadata.head(sample_size)
            
            # Property: Sample should preserve all characteristics
            sample_dx = set(sample_metadata['dx'].unique())
            assert sample_dx.issubset(expected_dx), "Sample diagnosis classes should be valid"
            
            # Property: Label encoding should work on sample
            encoded = loader.encode_labels(sample_metadata['dx'])
            assert len(encoded) == len(sample_metadata), "Encoded labels should match sample size"
            assert all(label >= 0 for label in encoded), "All sample labels should be encodable"
            
        except FileNotFoundError:
            # Skip test if real dataset is not available
            pytest.skip("Real HAM10000 dataset not available for testing")


if __name__ == "__main__":
    # Run property tests
    pytest.main([__file__, "-v"])