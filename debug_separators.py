#!/usr/bin/env python3
"""
Debug script to visualize detected separators
"""
import cv2
import numpy as np
import sys
import os
sys.path.append('.')

from card_detector import CardDetector
import logging

# Set up logging to see debug messages
logging.basicConfig(level=logging.DEBUG)

def debug_separators():
    """Debug what separators we're actually detecting"""
    
    # Load the debug PDF page we extracted
    image_path = "./debug_pdf_page.png"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
        
    print(f"Debugging separator detection on: {image_path}")
    print(f"Image size: {image.shape}")
    
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    detector = CardDetector()
    
    # Get the separators
    horizontal_lines = detector._find_grid_separators(gray, 'horizontal')
    vertical_lines = detector._find_grid_separators(gray, 'vertical')
    
    print(f"Found horizontal separators: {horizontal_lines}")
    print(f"Found vertical separators: {vertical_lines}")
    
    # Create visualization
    debug_image = image.copy()
    
    # Draw horizontal separators in red
    for y in horizontal_lines:
        cv2.line(debug_image, (0, y), (width, y), (0, 0, 255), 3)
        cv2.putText(debug_image, f"H{y}", (10, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Draw vertical separators in blue
    for x in vertical_lines:
        cv2.line(debug_image, (x, 0), (x, height), (255, 0, 0), 3)
        cv2.putText(debug_image, f"V{x}", (x+5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Expected separators for perfect 3x3 grid
    expected_h1 = height // 3
    expected_h2 = 2 * height // 3
    expected_v1 = width // 3
    expected_v2 = 2 * width // 3
    
    # Draw expected separators in green (dashed effect)
    for y in [expected_h1, expected_h2]:
        for x in range(0, width, 20):
            cv2.line(debug_image, (x, y), (x+10, y), (0, 255, 0), 2)
    
    for x in [expected_v1, expected_v2]:
        for y in range(0, height, 20):
            cv2.line(debug_image, (x, y), (x, y+10), (0, 255, 0), 2)
    
    print(f"Expected separators for 3x3 grid:")
    print(f"  Horizontal: {expected_h1}, {expected_h2}")
    print(f"  Vertical: {expected_v1}, {expected_v2}")
    
    # Save visualization
    cv2.imwrite("./debug_separators_visualization.png", debug_image)
    print(f"Saved separator visualization to: ./debug_separators_visualization.png")
    print("Red lines = detected horizontal separators")
    print("Blue lines = detected vertical separators") 
    print("Green dashed lines = expected 3x3 grid separators")

if __name__ == "__main__":
    debug_separators()