#!/usr/bin/env python3
"""
Quick OCR test - process all PDFs and show just the extracted text
"""
import sys
import os
import glob

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pdf_processor import PDFProcessor
from card_detector import CardDetector
from ocr_extractor import MTGTextExtractor

def quick_test_all_pdfs():
    """Test OCR on all PDFs in input directory"""
    
    pdf_dir = "./data/input_pdfs"
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return
    
    print(f"Testing OCR on {len(pdf_files)} PDFs")
    print("=" * 80)
    
    # Initialize processors
    pdf_processor = PDFProcessor()
    card_detector = CardDetector()
    ocr_extractor = MTGTextExtractor()
    
    for pdf_path in pdf_files:
        pdf_name = os.path.basename(pdf_path)
        print(f"\n{pdf_name}")
        print("-" * 60)
        
        try:
            # Process PDF
            pdf_type = pdf_processor.detect_pdf_type(pdf_path)
            images = pdf_processor.convert_pdf_to_images(pdf_path, pdf_type)
            
            if not images:
                print("Failed to convert PDF")
                continue
            
            page_image = images[0]
            print(f"PDF converted: {page_image.shape[1]}x{page_image.shape[0]} pixels")
            
            # Detect cards
            cards = card_detector.detect_card_grid_advanced(page_image)
            print(f"Cards detected: {len(cards)}")
            
            if not cards:
                print("No cards detected!")
                continue
            
            # Quick OCR on each card
            for i, card_image in enumerate(cards, 1):
                card_data = ocr_extractor.extract_card_attributes(card_image)
                name = card_data.get('name', 'NONE')
                confidence = card_data.get('confidence', 0)
                all_text = card_data.get('all_text', '')[:80]  # First 80 chars
                
                print(f"Card {i:2d}: '{name}' (conf: {confidence:.2f}) - {all_text}...")
                
        except Exception as e:
            print(f"Error processing {pdf_name}: {str(e)}")

if __name__ == "__main__":
    quick_test_all_pdfs()