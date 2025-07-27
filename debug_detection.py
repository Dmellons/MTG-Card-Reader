#!/usr/bin/env python3
"""
Debug script to visualize what's happening with card detection
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

def debug_detection_on_pdf():
    """Debug detection on a sample PDF page"""
    # Look for a processed PDF image in the temp/cache directory
    pdf_dirs = [
        "./temp",
        "./cache", 
        "./data",
        "./data/temp"
    ]
    
    pdf_image = None
    pdf_path = None
    
    # Try to find a PDF page image
    for dir_path in pdf_dirs:
        if os.path.exists(dir_path):
            for file in os.listdir(dir_path):
                if file.endswith(('.png', '.jpg', '.jpeg')) and 'page' in file.lower():
                    pdf_path = os.path.join(dir_path, file)
                    pdf_image = cv2.imread(pdf_path)
                    if pdf_image is not None:
                        break
            if pdf_image is not None:
                break
    
    if pdf_image is None:
        print("No PDF page image found. Please run a batch first to generate temp images.")
        return
        
    print(f"Found PDF image: {pdf_path}")
    print(f"Image size: {pdf_image.shape}")
    
    # Test all detection methods
    detector = CardDetector()
    
    print("\n=== Testing Border Detection ===")
    border_cards = detector.detect_cards_by_borders(pdf_image)
    print(f"Border detection found: {len(border_cards)} cards")
    
    print("\n=== Testing Shape Detection ===")
    shape_cards = detector.detect_cards_by_shape(pdf_image)
    print(f"Shape detection found: {len(shape_cards)} cards")
    
    print("\n=== Testing Advanced Grid Detection ===")
    grid_cards = detector.detect_card_grid_advanced(pdf_image)
    print(f"Grid detection found: {len(grid_cards)} cards")
    
    # Save a visualization
    debug_image = pdf_image.copy()
    
    # Draw rectangles around detected regions for visualization
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Green, Blue, Red
    methods = [border_cards, shape_cards, grid_cards]
    method_names = ["Border", "Shape", "Grid"]
    
    for i, (cards, color, name) in enumerate(zip(methods, colors, method_names)):
        for j, card in enumerate(cards):
            # For visualization, we'd need to track the original positions
            # This is a limitation of the current design
            print(f"{name} method card {j+1}: {card.shape}")
    
    # Save the original image for inspection
    cv2.imwrite("./debug_original_pdf_page.png", pdf_image)
    print(f"\nSaved original PDF page as: ./debug_original_pdf_page.png")
    print("Please inspect this image to see what the detection algorithm is working with.")

if __name__ == "__main__":
    debug_detection_on_pdf()