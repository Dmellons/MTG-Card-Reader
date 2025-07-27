#!/usr/bin/env python3
"""
Test script for the new precise 3x3 detection
"""
import cv2
import sys
import os
sys.path.append('.')

from card_detector import CardDetector
import logging

# Set up logging to see debug messages
logging.basicConfig(level=logging.DEBUG)

def test_precise_detection():
    """Test the new precise 3x3 detection method"""
    
    # Load the debug PDF page we extracted
    image_path = "./debug_pdf_page.png"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
        
    print(f"Testing precise 3x3 detection on: {image_path}")
    print(f"Image size: {image.shape}")
    
    detector = CardDetector()
    
    # Test the new precise detection method directly
    print("\n=== Testing Precise 3x3 Detection ===")
    precise_cards = detector._detect_precise_3x3_grid(image)
    print(f"Precise 3x3 detection found: {len(precise_cards)} cards")
    
    # Test through the main detection method
    print("\n=== Testing Main Detection with 3x3 ===")
    main_cards = detector.detect_card_grid_advanced(image, 3, 3)
    print(f"Main detection found: {len(main_cards)} cards")
    
    # Save the results
    for i, card in enumerate(precise_cards[:9]):
        output_path = f"./debug_precise_card_{i+1}.png"
        cv2.imwrite(output_path, card)
        print(f"Saved card {i+1}: {card.shape} -> {output_path}")
    
    print(f"\nSaved first {len(precise_cards)} cards from precise detection")
    print("Compare these with the original problematic extractions")

if __name__ == "__main__":
    test_precise_detection()