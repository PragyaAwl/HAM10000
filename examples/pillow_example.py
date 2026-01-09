"""Example of how Pillow (PIL) will be used in HAM10000 QSPICE Pipeline."""

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import os
from pathlib import Path

# Example 1: Loading HAM10000 images (Task 2)
def load_ham10000_image(image_path):
    """Load and validate HAM10000 image files."""
    try:
        # Load image
        image = Image.open(image_path)
        
        # Convert to RGB if needed (some images might be grayscale or RGBA)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        print(f"Loaded image: {image_path}")
        print(f"  Size: {image.size}")
        print(f"  Mode: {image.mode}")
        print(f"  Format: {image.format}")
        
        return image
    
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

# Example 2: Image preprocessing and resizing (Task 2)
def preprocess_ham10000_image(image, target_size=(224, 224)):
    """Preprocess HAM10000 images for EfficientNet input."""
    # Resize image while maintaining aspect ratio
    image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Normalize to [0, 1] range
    image_array = image_array.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_array = (image_array - mean) / std
    
    # Convert to CHW format (Channel, Height, Width)
    image_array = np.transpose(image_array, (2, 0, 1))
    
    return image_array

# Example 3: Data augmentation for training (Task 2)
def augment_ham10000_image(image):
    """Apply data augmentation to HAM10000 images."""
    augmented_images = []
    
    # Original image
    augmented_images.append(("original", image))
    
    # Rotation (skin lesions can appear at any angle)
    rotated = image.rotate(45, expand=True, fillcolor=(255, 255, 255))
    augmented_images.append(("rotated_45", rotated))
    
    # Horizontal flip (lesions can appear on either side)
    flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
    augmented_images.append(("flipped", flipped))
    
    # Brightness adjustment (different lighting conditions)
    enhancer = ImageEnhance.Brightness(image)
    brighter = enhancer.enhance(1.3)  # 30% brighter
    darker = enhancer.enhance(0.7)    # 30% darker
    augmented_images.append(("brighter", brighter))
    augmented_images.append(("darker", darker))
    
    # Contrast adjustment (different camera settings)
    enhancer = ImageEnhance.Contrast(image)
    high_contrast = enhancer.enhance(1.3)
    low_contrast = enhancer.enhance(0.7)
    augmented_images.append(("high_contrast", high_contrast))
    augmented_images.append(("low_contrast", low_contrast))
    
    # Color saturation (different skin tones)
    enhancer = ImageEnhance.Color(image)
    saturated = enhancer.enhance(1.2)
    desaturated = enhancer.enhance(0.8)
    augmented_images.append(("saturated", saturated))
    augmented_images.append(("desaturated", desaturated))
    
    # Slight blur (camera focus issues)
    blurred = image.filter(ImageFilter.GaussianBlur(radius=0.5))
    augmented_images.append(("blurred", blurred))
    
    return augmented_images

# Example 4: Image quality validation (Task 2)
def validate_image_quality(image, min_size=(100, 100), max_size=(2000, 2000)):
    """Validate HAM10000 image quality and detect corrupted files."""
    issues = []
    
    # Check image size
    width, height = image.size
    if width < min_size[0] or height < min_size[1]:
        issues.append(f"Image too small: {width}x{height} < {min_size}")
    
    if width > max_size[0] or height > max_size[1]:
        issues.append(f"Image too large: {width}x{height} > {max_size}")
    
    # Check aspect ratio (skin lesion images shouldn't be extremely elongated)
    aspect_ratio = max(width, height) / min(width, height)
    if aspect_ratio > 3.0:
        issues.append(f"Unusual aspect ratio: {aspect_ratio:.2f}")
    
    # Check if image is mostly black or white (corrupted/empty images)
    image_array = np.array(image)
    mean_intensity = np.mean(image_array)
    if mean_intensity < 10:
        issues.append("Image appears to be mostly black")
    elif mean_intensity > 245:
        issues.append("Image appears to be mostly white")
    
    # Check color channels (should be RGB)
    if len(image_array.shape) != 3 or image_array.shape[2] != 3:
        issues.append(f"Unexpected image shape: {image_array.shape}")
    
    return len(issues) == 0, issues

# Example 5: Batch image processing (Task 2)
def process_ham10000_batch(image_directory, output_directory):
    """Process a batch of HAM10000 images."""
    image_dir = Path(image_directory)
    output_dir = Path(output_directory)
    output_dir.mkdir(exist_ok=True)
    
    # Supported image formats
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    processed_count = 0
    error_count = 0
    
    for image_path in image_dir.iterdir():
        if image_path.suffix.lower() in supported_formats:
            try:
                # Load image
                image = load_ham10000_image(image_path)
                if image is None:
                    error_count += 1
                    continue
                
                # Validate quality
                is_valid, issues = validate_image_quality(image)
                if not is_valid:
                    print(f"Quality issues in {image_path.name}: {issues}")
                    error_count += 1
                    continue
                
                # Preprocess
                processed_array = preprocess_ham10000_image(image)
                
                # Save processed image (as numpy array)
                output_path = output_dir / f"{image_path.stem}_processed.npy"
                np.save(output_path, processed_array)
                
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} images...")
                
            except Exception as e:
                print(f"Error processing {image_path.name}: {e}")
                error_count += 1
    
    print(f"\nBatch processing complete:")
    print(f"  Successfully processed: {processed_count}")
    print(f"  Errors: {error_count}")
    
    return processed_count, error_count

# Example 6: Creating image thumbnails for visualization (Task 9)
def create_result_visualization(original_image, sram_image, predictions):
    """Create visualization comparing original vs SRAM model results."""
    # Create thumbnail versions
    thumb_size = (150, 150)
    original_thumb = original_image.copy()
    original_thumb.thumbnail(thumb_size, Image.Resampling.LANCZOS)
    
    sram_thumb = sram_image.copy()
    sram_thumb.thumbnail(thumb_size, Image.Resampling.LANCZOS)
    
    # Create side-by-side comparison
    comparison_width = thumb_size[0] * 2 + 20  # 20px gap
    comparison_height = thumb_size[1] + 100    # 100px for text
    
    comparison = Image.new('RGB', (comparison_width, comparison_height), 'white')
    
    # Paste thumbnails
    comparison.paste(original_thumb, (0, 50))
    comparison.paste(sram_thumb, (thumb_size[0] + 20, 50))
    
    # In a real implementation, you'd add text labels here
    # This would require additional libraries like PIL.ImageDraw
    
    return comparison

if __name__ == "__main__":
    print("=== HAM10000 Pillow (PIL) Usage Examples ===\n")
    
    # Create a sample image for demonstration
    sample_image = Image.new('RGB', (300, 300), color='lightblue')
    
    print("1. Image Loading and Preprocessing:")
    processed = preprocess_ham10000_image(sample_image)
    print(f"Processed image shape: {processed.shape}")
    
    print("\n2. Data Augmentation:")
    augmented = augment_ham10000_image(sample_image)
    print(f"Generated {len(augmented)} augmented versions")
    
    print("\n3. Image Quality Validation:")
    is_valid, issues = validate_image_quality(sample_image)
    print(f"Image valid: {is_valid}")
    if issues:
        print(f"Issues: {issues}")
    
    print("\n4. Visualization:")
    comparison = create_result_visualization(sample_image, sample_image, {})
    print(f"Created comparison image: {comparison.size}")