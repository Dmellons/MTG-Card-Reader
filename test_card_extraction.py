#!/usr/bin/env python3
"""
Test script for card extraction and OCR only
Shows what text is being extracted from each card before identification
"""
import sys
import os
import cv2
import logging
import warnings
from pathlib import Path

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pdf_processor import PDFProcessor
from card_detector import CardDetector
from ocr_extractor import MTGTextExtractor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_card_extraction_and_ocr(pdf_path: str):
    """Test card extraction and OCR on a single PDF"""
    
    print(f"\nTesting card extraction on: {pdf_path}")
    print("=" * 60)
    
    try:
        # Initialize processors
        pdf_processor = PDFProcessor()
        card_detector = CardDetector()
        ocr_extractor = MTGTextExtractor()
        
        # Process PDF
        print("Converting PDF to image...")
        pdf_type = pdf_processor.detect_pdf_type(pdf_path)
        print(f"PDF type detected: {pdf_type}")
        
        images = pdf_processor.convert_pdf_to_images(pdf_path, pdf_type)
        
        if not images:
            print("Failed to convert PDF to images")
            return
        
        page_image = images[0]
        print(f"PDF converted: {page_image.shape[1]}x{page_image.shape[0]} pixels")
        
        # Detect cards
        print("\nDetecting cards...")
        cards = card_detector.detect_card_grid_advanced(page_image)
        print(f"Found {len(cards)} cards")
        
        if not cards:
            print("No cards detected!")
            return
        
        # Process each card with OCR
        print(f"\nExtracting text from each card:")
        print("-" * 60)
        
        for i, card_image in enumerate(cards, 1):
            print(f"\nCard {i}: {card_image.shape[1]}x{card_image.shape[0]} pixels")
            
            # Extract text attributes
            card_data = ocr_extractor.extract_card_attributes(card_image)
            
            print(f"  Name: '{card_data.get('name', 'None')}'")
            print(f"  Mana Cost: '{card_data.get('mana_cost', 'None')}'")
            print(f"  Type Line: '{card_data.get('type_line', 'None')}'")
            print(f"  Oracle Text: '{card_data.get('oracle_text', 'None')}'")
            print(f"  Power/Toughness: '{card_data.get('power_toughness', 'None')}'")
            print(f"  Confidence: {card_data.get('confidence', 0):.2f}")
            
            # Show all extracted text
            all_text = card_data.get('all_text', '').strip()
            if all_text:
                print(f"  All Text: '{all_text[:100]}{'...' if len(all_text) > 100 else ''}'")
            
            # Save debug image
            debug_dir = "./data/debug_cards"
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, f"test_card_{i:02d}.png")
            cv2.imwrite(debug_path, card_image)
            print(f"  Saved debug image: {debug_path}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python test_card_extraction.py <pdf_path>")
        print("Example: python test_card_extraction.py ./data/input_pdfs/scan.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        sys.exit(1)
    
    test_card_extraction_and_ocr(pdf_path)
    
    print(f"\nTest complete! Check ./data/debug_cards/ for extracted card images")

if __name__ == "__main__":
    main()