#!/usr/bin/env python3
"""
Simple test script to verify the optimized lego art conversion
and compare performance.
"""
import cv2
import numpy as np
import csv
import time
import math

lego_unit_size = 8  # mm


class LegoArtConverter:
    """Optimized LEGO art converter for testing"""
    
    def __init__(self):
        self.colour_list = []
        self.load_csv()
    
    def load_csv(self):
        with open('colors.csv', newline='') as csvfile:
            dict_reader = csv.DictReader(csvfile)
            for line in dict_reader:
                self.colour_list.append(line)
        
        for c in self.colour_list:
            c['rgb_array'] = bytearray.fromhex(c['rgb'])

            single_pixel_image = np.zeros((1, 1, 3), np.uint8)
            single_pixel_image[0, 0] = c['rgb_array']

            out_lab = cv2.cvtColor(single_pixel_image, cv2.COLOR_RGB2LAB)
            c['lab_array'] = out_lab[0, 0]
            c['lab_array_norm'] = self.lab_normalization(c['lab_array'])
        
        # Pre-compute arrays for vectorized operations
        self.lab_array_norm_matrix = np.array([c['lab_array_norm'] for c in self.colour_list])
        self.lab_array_matrix = np.array([c['lab_array'] for c in self.colour_list])

    def lab_normalization(self, arr1):
        out = []
        out.append(arr1[0] * 100 / 256)
        out.append(arr1[1] - 128)
        out.append(arr1[2] - 128)
        return out
    
    def convert_image_to_lego_colours(self, image):
        """Optimized version using vectorized NumPy operations"""
        self.lego_pieces = {}
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        height, width, depth = lab_image.shape

        # Normalize the entire LAB image at once
        lab_image_norm = np.zeros((height, width, 3), dtype=np.float32)
        lab_image_norm[:, :, 0] = lab_image[:, :, 0] * 100 / 256
        lab_image_norm[:, :, 1] = lab_image[:, :, 1] - 128
        lab_image_norm[:, :, 2] = lab_image[:, :, 2] - 128
        
        # Reshape for vectorized computation: (height*width, 3)
        pixels_flat = lab_image_norm.reshape(-1, 3)
        
        # Compute distances from all pixels to all colors using broadcasting
        distances = np.sqrt(np.sum((pixels_flat[:, np.newaxis, :] - 
                                   self.lab_array_norm_matrix[np.newaxis, :, :]) ** 2, axis=2))
        
        # Find the index of the closest color for each pixel
        closest_color_indices = np.argmin(distances, axis=1)
        
        # Map each pixel to its closest LAB color
        lab_image_flat = self.lab_array_matrix[closest_color_indices]
        lab_image = lab_image_flat.reshape(height, width, 3).astype(np.uint8)
        
        # Count occurrences of each color
        unique_indices, counts = np.unique(closest_color_indices, return_counts=True)
        
        for idx, count in zip(unique_indices, counts):
            colour = self.colour_list[idx]
            if colour['id'] not in self.lego_pieces:
                self.lego_pieces[colour['id']] = colour.copy()
                self.lego_pieces[colour['id']]['count'] = 0
            self.lego_pieces[colour['id']]['count'] += count

        return cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)


def test_basic_conversion():
    """Test that the conversion works correctly"""
    print("Testing basic image conversion...")
    
    converter = LegoArtConverter()
    
    # Load test image
    input_image = cv2.imread("turtle.jpg")
    if input_image is None:
        print("ERROR: Could not load turtle.jpg")
        return False
    
    print(f"Loaded image with shape: {input_image.shape}")
    
    # Resize to a small test size
    test_image = cv2.resize(input_image, (50, 50), interpolation=cv2.INTER_CUBIC)
    
    # Convert to lego colors
    start_time = time.time()
    result = converter.convert_image_to_lego_colours(test_image)
    elapsed = time.time() - start_time
    
    print(f"Conversion completed in {elapsed:.4f} seconds")
    print(f"Result shape: {result.shape}")
    print(f"Number of unique lego colors used: {len(converter.lego_pieces)}")
    
    # Verify result has correct shape
    assert result.shape == test_image.shape, "Output shape doesn't match input shape"
    
    # Verify result is not all zeros
    assert np.any(result), "Result image is all zeros"
    
    print("✓ Basic conversion test PASSED")
    return True


def benchmark_performance():
    """Benchmark the optimized conversion"""
    print("\n" + "="*60)
    print("Performance Benchmark")
    print("="*60)
    
    converter = LegoArtConverter()
    
    # Load test image
    input_image = cv2.imread("turtle.jpg")
    if input_image is None:
        print("ERROR: Could not load turtle.jpg")
        return
    
    # Test different sizes
    test_sizes = [
        (20, 20),
        (50, 50),
        (100, 100),
    ]
    
    for size in test_sizes:
        test_image = cv2.resize(input_image, size, interpolation=cv2.INTER_CUBIC)
        
        # Run conversion
        start_time = time.time()
        result = converter.convert_image_to_lego_colours(test_image)
        elapsed = time.time() - start_time
        
        pixels = size[0] * size[1]
        colors_used = len(converter.lego_pieces)
        
        print(f"\nSize: {size[0]}x{size[1]} ({pixels:,} pixels)")
        print(f"  Time: {elapsed:.4f} seconds")
        print(f"  Speed: {pixels/elapsed:.0f} pixels/second")
        print(f"  Colors used: {colors_used}")


if __name__ == "__main__":
    print("LEGO Art Converter - Performance Test")
    print("="*60)
    
    # Run tests
    if test_basic_conversion():
        benchmark_performance()
        print("\n" + "="*60)
        print("All tests completed successfully!")
        print("="*60)
    else:
        print("\nTests FAILED")
