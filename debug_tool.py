#!/usr/bin/env python3
"""
Debug Tool for MTG Card Processing System
Helps diagnose issues with card detection and OCR
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pdf_processor import PDFProcessor, ImagePreprocessor
from card_detector import CardDetector, GridLayoutAnalyzer
from ocr_extractor import MTGTextExtractor
from config import settings

class DebugTool:
    """Debug tool for diagnosing processing issues"""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.card_detector = CardDetector()
        self.text_extractor = MTGTextExtractor()
    
    def debug_pdf(self, pdf_path: str, save_debug_images: bool = True):
        """Debug a PDF file step by step"""
        print(f"[DEBUG] Debugging PDF: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            print(f"[ERROR] PDF file not found: {pdf_path}")
            return
        
        # Step 1: Analyze PDF
        print("\n[PAGE] Step 1: PDF Analysis")
        pdf_type = self.pdf_processor.detect_pdf_type(pdf_path)
        print(f"   PDF Type: {pdf_type}")
        
        # Step 2: Convert to images
        print("\n[IMAGE] Step 2: Image Conversion")
        images = self.pdf_processor.convert_pdf_to_images(pdf_path, pdf_type)
        print(f"   Pages converted: {len(images)}")
        
        if not images:
            print("[ERROR] No images extracted from PDF")
            return
        
        # Process each page
        for page_idx, page_image in enumerate(images):
            print(f"\n[PAGE] Page {page_idx + 1}:")
            self.debug_page(page_image, page_idx, save_debug_images, pdf_path)
    
    def debug_page(self, page_image: np.ndarray, page_idx: int, 
                   save_images: bool = True, pdf_name: str = ""):
        """Debug a single page"""
        
        # Save original page if requested
        if save_images:
            debug_dir = "./debug_output"
            os.makedirs(debug_dir, exist_ok=True)
            
            page_filename = f"page_{page_idx+1:03d}_original.png"
            if pdf_name:
                base_name = Path(pdf_name).stem
                page_filename = f"{base_name}_page_{page_idx+1:03d}_original.png"
            
            cv2.imwrite(os.path.join(debug_dir, page_filename), page_image)
            print(f"   [SAVED] Saved: {page_filename}")
        
        # Step 3: Layout Analysis
        print("   [DEBUG] Layout Analysis:")
        layout = GridLayoutAnalyzer.analyze_page_layout(page_image)
        print(f"      Suggested grid: {layout['suggested_rows']}x{layout['suggested_cols']}")
        print(f"      Confidence: {layout['confidence']:.2f}")
        print(f"      H peaks: {layout['h_peaks']}, V peaks: {layout['v_peaks']}")
        
        # Step 4: Card Detection
        print("   [DETECT] Card Detection:")
        detected_cards = self.card_detector.detect_card_grid_advanced(
            page_image, layout['suggested_rows'], layout['suggested_cols']
        )
        print(f"      Cards detected: {len(detected_cards)}")
        
        if len(detected_cards) == 0:
            # Try alternative detection methods
            print("   [RETRY] Trying alternative detection...")
            
            # Try different grid sizes
            for rows in [2, 3, 4, 5]:
                for cols in [2, 3, 4, 5]:
                    cards = self.card_detector.detect_card_grid_advanced(page_image, rows, cols)
                    if len(cards) > 0:
                        print(f"      [SUCCESS] Found {len(cards)} cards with {rows}x{cols} grid")
                        detected_cards = cards
                        break
                if len(detected_cards) > 0:
                    break
        
        # Step 5: Save detected cards and test OCR
        if detected_cards and save_images:
            print("   [OCR] Testing OCR on detected cards:")
            
            for i, card_image in enumerate(detected_cards[:5]):  # Test first 5 cards
                # Save card image
                card_filename = f"page_{page_idx+1:03d}_card_{i+1:02d}.png"
                if pdf_name:
                    base_name = Path(pdf_name).stem
                    card_filename = f"{base_name}_page_{page_idx+1:03d}_card_{i+1:02d}.png"
                
                card_path = os.path.join(debug_dir, card_filename)
                cv2.imwrite(card_path, card_image)
                
                # Test OCR
                print(f"      Card {i+1}: {card_image.shape} -> ", end="")
                
                try:
                    ocr_results = self.text_extractor.extract_card_attributes(card_image)
                    if ocr_results and ocr_results.get('name'):
                        print(f"'{ocr_results['name']}' (conf: {ocr_results.get('confidence', 0):.2f})")
                    elif ocr_results and ocr_results.get('all_text'):
                        text_preview = ocr_results['all_text'][:50] + "..." if len(ocr_results['all_text']) > 50 else ocr_results['all_text']
                        print(f"Text: '{text_preview}'")
                    else:
                        print("No text detected")
                except Exception as e:
                    print(f"OCR Error: {str(e)}")
    
    def debug_single_image(self, image_path: str):
        """Debug a single card image"""
        print(f"[DEBUG] Debugging image: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"[ERROR] Image file not found: {image_path}")
            return
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Could not load image: {image_path}")
            return
        
        print(f"   [INFO] Image size: {image.shape}")
        
        # Test OCR
        print("   [OCR] Testing OCR:")
        try:
            ocr_results = self.text_extractor.extract_card_attributes(image)
            
            if ocr_results:
                print(f"      Name: {ocr_results.get('name', 'Not found')}")
                print(f"      Mana Cost: {ocr_results.get('mana_cost', 'Not found')}")
                print(f"      Type: {ocr_results.get('type_line', 'Not found')}")
                print(f"      Confidence: {ocr_results.get('confidence', 0):.2f}")
                
                if ocr_results.get('all_text'):
                    print(f"      All text: {ocr_results['all_text'][:100]}...")
            else:
                print("      No OCR results")
        
        except Exception as e:
            print(f"      OCR Error: {str(e)}")
    
    def test_ocr_engines(self, image_path: str):
        """Test different OCR engines on an image"""
        print(f"[TEST] Testing OCR engines on: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"[ERROR] Image file not found: {image_path}")
            return
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Could not load image: {image_path}")
            return
        
        # Test EasyOCR
        print("\n[RESULTS] EasyOCR Results:")
        try:
            extractor_easy = MTGTextExtractor(ocr_engine='easyocr')
            results_easy = extractor_easy.extract_card_attributes(image)
            if results_easy:
                print(f"   Name: {results_easy.get('name', 'None')}")
                print(f"   Text: {results_easy.get('all_text', 'None')[:100]}...")
            else:
                print("   No results")
        except Exception as e:
            print(f"   Error: {str(e)}")
        
        # Test Tesseract
        print("\n[RESULTS] Tesseract Results:")
        try:
            extractor_tess = MTGTextExtractor(ocr_engine='tesseract')
            results_tess = extractor_tess.extract_card_attributes(image)
            if results_tess:
                print(f"   Name: {results_tess.get('name', 'None')}")
                print(f"   Text: {results_tess.get('all_text', 'None')[:100]}...")
            else:
                print("   No results")
        except Exception as e:
            print(f"   Error: {str(e)}")
    
    def visualize_detection(self, pdf_path: str, page_num: int = 0):
        """Create visualization of card detection process"""
        print(f"[VIS] Creating detection visualization for: {pdf_path}")
        
        # Convert PDF to image
        images = self.pdf_processor.convert_pdf_to_images(pdf_path)
        if not images or page_num >= len(images):
            print(f"[ERROR] Could not get page {page_num}")
            return
        
        page_image = images[page_num]
        
        # Analyze layout
        layout = GridLayoutAnalyzer.analyze_page_layout(page_image)
        
        # Create visualization
        vis_image = page_image.copy()
        
        # Draw detected grid lines (simplified visualization)
        gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Detect and draw card boundaries
        detected_cards = self.card_detector.detect_card_grid_advanced(
            page_image, layout['suggested_rows'], layout['suggested_cols']
        )
        
        # Save visualization
        debug_dir = "./debug_output"
        os.makedirs(debug_dir, exist_ok=True)
        
        vis_filename = f"detection_visualization_page_{page_num+1}.png"
        cv2.imwrite(os.path.join(debug_dir, vis_filename), vis_image)
        
        print(f"   [SAVED] Saved visualization: {vis_filename}")
        print(f"   [DETECT] Detected {len(detected_cards)} cards")

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MTG Card Processing Debug Tool')
    parser.add_argument('action', choices=['pdf', 'image', 'ocr', 'visualize'], 
                       help='Action to perform')
    parser.add_argument('file_path', help='Path to PDF or image file')
    parser.add_argument('--page', type=int, default=0, help='Page number for visualization')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save debug images')
    
    args = parser.parse_args()
    
    debug_tool = DebugTool()
    
    if args.action == 'pdf':
        debug_tool.debug_pdf(args.file_path, not args.no_save)
    elif args.action == 'image':
        debug_tool.debug_single_image(args.file_path)
    elif args.action == 'ocr':
        debug_tool.test_ocr_engines(args.file_path)
    elif args.action == 'visualize':
        debug_tool.visualize_detection(args.file_path, args.page)

if __name__ == "__main__":
    main()