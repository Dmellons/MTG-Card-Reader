#!/usr/bin/env python3
"""
Test script to verify the fixes for OCR and card identification issues.
"""

import sys
import os
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from main_processor import MTGCardProcessingSystem
from card_identifier import MTGCardIdentifier, HybridIdentifier
from ocr_extractor import MTGTextExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_null_safety():
    """Test that None values are handled safely"""
    print("=" * 50)
    print("Testing Null Safety Fixes")
    print("=" * 50)
    
    # Test MTGCardIdentifier with None values
    identifier = MTGCardIdentifier()
    
    # These should not crash
    result1 = identifier.identify_by_name(None)
    result2 = identifier.identify_by_name("")
    result3 = identifier.identify_by_name("  ")
    result4 = identifier.identify_by_name("A")  # Too short
    
    print(f"✓ identify_by_name(None): {result1}")
    print(f"✓ identify_by_name(''): {result2}")
    print(f"✓ identify_by_name('  '): {result3}")
    print(f"✓ identify_by_name('A'): {result4}")
    
    # Test HybridIdentifier
    hybrid = HybridIdentifier()
    
    # Test with None extracted_name (this was causing the original error)
    import numpy as np
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    result5 = hybrid.identify_card_comprehensive(dummy_image, None, None, None)
    print(f"✓ hybrid.identify_card_comprehensive with None name: {result5.get('success', False)}")
    
def test_ocr_extraction():
    """Test OCR extraction with no results"""
    print("\n" + "=" * 50)
    print("Testing OCR Extraction Improvements")
    print("=" * 50)
    
    extractor = MTGTextExtractor()
    
    # Test with minimal/blank image
    import numpy as np
    blank_image = np.ones((200, 150, 3), dtype=np.uint8) * 255  # White image
    
    try:
        result = extractor.extract_card_attributes(blank_image)
        print(f"✓ Blank image extraction result: {result}")
        print(f"  - Name: {result.get('name')}")
        print(f"  - Confidence: {result.get('confidence', 0)}")
        print(f"  - All text: '{result.get('all_text', '')}'")
    except Exception as e:
        print(f"✗ OCR extraction failed: {e}")
        return False
    
    return True

def test_processing_system():
    """Test the main processing system with error handling"""
    print("\n" + "=" * 50)
    print("Testing Main Processing System")
    print("=" * 50)
    
    try:
        # Check if sample images exist
        sample_images_dir = project_root / "data" / "card_images"
        if sample_images_dir.exists():
            images = list(sample_images_dir.glob("*.png"))
            if images:
                print(f"✓ Found {len(images)} sample images")
                
                processor = MTGCardProcessingSystem()
                
                # Test processing a single image
                import cv2
                test_image = cv2.imread(str(images[0]))
                if test_image is not None:
                    result = processor.process_single_card_image(str(images[0]))
                    print(f"✓ Single image processing: {result.get('success', False)}")
                else:
                    print("✗ Could not load test image")
            else:
                print("! No sample images found for testing")
        else:
            print("! Sample images directory not found")
    except Exception as e:
        print(f"✗ Processing system test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("MTG Card Identifier - Testing Fixes")
    print("===================================")
    
    all_passed = True
    
    try:
        test_null_safety()
    except Exception as e:
        print(f"✗ Null safety tests failed: {e}")
        all_passed = False
    
    try:
        if not test_ocr_extraction():
            all_passed = False
    except Exception as e:
        print(f"✗ OCR tests failed: {e}")
        all_passed = False
    
    try:
        if not test_processing_system():
            all_passed = False
    except Exception as e:
        print(f"✗ Processing system tests failed: {e}")
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All tests passed! The fixes appear to be working.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("=" * 50)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)