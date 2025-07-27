#!/usr/bin/env python3
"""
Debug script to extract and save PDF pages for inspection
"""
import cv2
import sys
import os
sys.path.append('.')

from pdf_processor import PDFProcessor

def extract_pdf_page():
    """Extract first page from a sample PDF"""
    pdf_path = "./data/input_pdfs/2025-07-26 MTG_Scan_2025-07-26_105910.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"PDF not found: {pdf_path}")
        return
    
    print(f"Processing PDF: {pdf_path}")
    
    # Use the existing PDF processor
    processor = PDFProcessor()
    
    try:
        # Extract pages
        pages = processor.convert_pdf_to_images(pdf_path)
        
        if not pages:
            print("No pages extracted")
            return
            
        print(f"Extracted {len(pages)} pages")
        
        # Save first page for inspection
        first_page = pages[0]
        output_path = "./debug_pdf_page.png"
        cv2.imwrite(output_path, first_page)
        
        print(f"Saved page as: {output_path}")
        print(f"Page size: {first_page.shape}")
        
        # Now test detection on this page
        from card_detector import CardDetector
        detector = CardDetector()
        
        print("\n=== Testing Border Detection ===")
        border_cards = detector.detect_cards_by_borders(first_page)
        print(f"Border detection found: {len(border_cards)} cards")
        
        print("\n=== Testing Shape Detection ===")
        shape_cards = detector.detect_cards_by_shape(first_page)
        print(f"Shape detection found: {len(shape_cards)} cards")
        
        print("\n=== Testing Whitespace Detection ===")
        whitespace_cards = detector._detect_cards_by_whitespace(first_page, 4, 4)
        print(f"Whitespace detection found: {len(whitespace_cards)} cards")
        
        # Save first few detected cards for comparison
        for i, card in enumerate(border_cards[:3]):
            cv2.imwrite(f"./debug_border_card_{i+1}.png", card)
        
        print(f"\nSaved first 3 border-detected cards as debug_border_card_*.png")
        print("Compare these with your sample cards to see the quality difference")
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    extract_pdf_page()