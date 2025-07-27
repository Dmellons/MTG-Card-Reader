#!/usr/bin/env python3
"""
Simple test script to debug card detection issues
"""
import cv2
import sys
sys.path.append('.')

from card_detector import CardDetector
from ocr_extractor import MTGTextExtractor

def test_sample_card():
    """Test with the provided sample card"""
    # Load sample card
    sample_path = "./data/sample_cards/basic_land.jpg"
    try:
        image = cv2.imread(sample_path)
        if image is None:
            print(f"Could not load image: {sample_path}")
            return
        
        print(f"Loaded sample card: {image.shape}")
        
        # Test OCR on sample card
        extractor = MTGTextExtractor()
        card_data = extractor.extract_card_attributes(image)
        
        print("Extracted data:")
        for key, value in card_data.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Error testing sample card: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sample_card()