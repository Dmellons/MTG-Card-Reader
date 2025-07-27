"""
Main Processing System - Orchestrates the complete MTG card processing workflow
"""
import os
import logging
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import cv2
import numpy as np

from config import settings
from models import DatabaseManager, CardRepository, Card, ProcessingLog
from pdf_processor import PDFProcessor, ImagePreprocessor
from card_detector import CardDetector, GridLayoutAnalyzer
from ocr_extractor import MTGTextExtractor, OCRValidator
from card_identifier import HybridIdentifier, validate_identification_result
from duplicate_handler import DuplicateHandler
from export_manager import ExportManager

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.LOG_FILE) if settings.LOG_FILE else logging.NullHandler(),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class MTGCardProcessingSystem:
    """Main processing system that orchestrates the complete workflow"""
    
    def __init__(self):
        # Initialize database
        self.db_manager = DatabaseManager(
            settings.DATABASE_URL,
            settings.DATABASE_POOL_SIZE,
            settings.DATABASE_MAX_OVERFLOW
        )
        
        # Initialize processing components
        self.pdf_processor = PDFProcessor()
        self.card_detector = CardDetector()
        self.text_extractor = MTGTextExtractor()
        self.card_identifier = HybridIdentifier()
        self.duplicate_handler = DuplicateHandler()
        self.export_manager = ExportManager()
        
        # Processing statistics
        self.stats = {
            'total_pages': 0,
            'total_cards_detected': 0,
            'cards_identified': 0,
            'cards_failed': 0,
            'duplicates_found': 0,
            'processing_time': 0
        }
        
        # Create tables if they don't exist
        try:
            self.db_manager.create_tables()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            raise
    
    def process_pdf_collection(self, pdf_path: str, collection_name: str = None) -> Dict:
        """
        Process a complete PDF collection
        Main entry point for the processing workflow
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting processing of PDF: {pdf_path}")
            
            # Validate input
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            collection_name = collection_name or f"collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Step 1: Analyze PDF and extract images
            pdf_analysis = self._analyze_pdf(pdf_path)
            if not pdf_analysis['success']:
                return pdf_analysis
            
            # Step 2: Process each page
            all_card_data = []
            page_results = []
            
            for page_idx, page_image in enumerate(pdf_analysis['page_images']):
                page_result = self._process_page(
                    page_image, page_idx, pdf_path, collection_name
                )
                page_results.append(page_result)
                all_card_data.extend(page_result['cards'])
            
            # Step 3: Batch process and store in database
            if all_card_data:
                db_result = self._store_cards_batch(all_card_data, collection_name)
            else:
                db_result = {'new_cards': 0, 'duplicates': 0, 'errors': 0}
            
            # Step 4: Generate exports
            export_result = self._generate_exports(collection_name)
            
            # Calculate final statistics
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            final_result = {
                'success': True,
                'collection_name': collection_name,
                'pdf_file': pdf_path,
                'processing_time': processing_time,
                'pages_processed': len(pdf_analysis['page_images']),
                'cards_detected': len(all_card_data),
                'cards_stored': db_result['new_cards'],
                'duplicates_found': db_result['duplicates'],
                'errors': db_result['errors'],
                'exports_generated': export_result,
                'page_details': page_results
            }
            
            logger.info(f"Processing complete: {final_result['cards_stored']} cards stored, "
                       f"{final_result['duplicates_found']} duplicates found")
            
            return final_result
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                'success': False,
                'error': error_msg,
                'collection_name': collection_name,
                'pdf_file': pdf_path
            }
    
    def _analyze_pdf(self, pdf_path: str) -> Dict:
        """Analyze PDF and extract page images"""
        try:
            # Detect PDF type
            pdf_type = self.pdf_processor.detect_pdf_type(pdf_path)
            logger.info(f"PDF type detected: {pdf_type}")
            
            # Convert to images
            page_images = self.pdf_processor.convert_pdf_to_images(pdf_path, pdf_type)
            
            if not page_images:
                return {
                    'success': False,
                    'error': 'No images could be extracted from PDF'
                }
            
            # Apply perspective correction if needed for scanned PDFs
            if pdf_type == 'scanned' and settings.PERSPECTIVE_CORRECTION:
                corrected_images = []
                for img in page_images:
                    corrected = ImagePreprocessor.auto_perspective_correction(img)
                    corrected_images.append(corrected)
                page_images = corrected_images
            
            return {
                'success': True,
                'pdf_type': pdf_type,
                'page_images': page_images,
                'total_pages': len(page_images)
            }
            
        except Exception as e:
            logger.error(f"PDF analysis failed: {str(e)}")
            return {
                'success': False,
                'error': f"PDF analysis failed: {str(e)}"
            }
    
    def _process_page(self, page_image: np.ndarray, page_idx: int, 
                     pdf_path: str, collection_name: str) -> Dict:
        """Process a single page to extract cards"""
        try:
            logger.debug(f"Processing page {page_idx + 1}")
            
            # Analyze page layout
            layout_analysis = GridLayoutAnalyzer.analyze_page_layout(page_image)
            suggested_rows = layout_analysis['suggested_rows']
            suggested_cols = layout_analysis['suggested_cols']
            
            logger.debug(f"Page layout analysis: {suggested_rows}x{suggested_cols} grid "
                        f"(confidence: {layout_analysis['confidence']:.2f})")
            
            # Prefer configured grid size for known MTG layouts
            # If layout analyzer has low confidence, use configured defaults
            if layout_analysis['confidence'] < 0.8:
                suggested_rows = self.card_detector.expected_rows
                suggested_cols = self.card_detector.expected_cols
                logger.debug(f"Using configured grid size due to low confidence: {suggested_rows}x{suggested_cols}")
            
            # Detect cards using determined layout
            detected_cards = self.card_detector.detect_card_grid_advanced(
                page_image, suggested_rows, suggested_cols
            )
            
            if not detected_cards:
                logger.warning(f"No cards detected on page {page_idx + 1}")
                return {
                    'page_number': page_idx + 1,
                    'cards_detected': 0,
                    'cards': [],
                    'layout_analysis': layout_analysis
                }
            
            # Process each detected card
            processed_cards = []
            for card_idx, card_image in enumerate(detected_cards):
                card_result = self._process_individual_card(
                    card_image, page_idx, card_idx, pdf_path, collection_name
                )
                
                if card_result:
                    processed_cards.append(card_result)
            
            return {
                'page_number': page_idx + 1,
                'cards_detected': len(detected_cards),
                'cards_processed': len(processed_cards),
                'cards': processed_cards,
                'layout_analysis': layout_analysis
            }
            
        except Exception as e:
            logger.error(f"Page processing failed for page {page_idx + 1}: {str(e)}")
            return {
                'page_number': page_idx + 1,
                'cards_detected': 0,
                'cards': [],
                'error': str(e)
            }
    
    def _process_individual_card(self, card_image: np.ndarray, page_idx: int, 
                               card_idx: int, pdf_path: str, collection_name: str) -> Optional[Dict]:
        """Process a single card image"""
        try:
            # Validate card image
            if not self._validate_card_image(card_image):
                logger.warning(f"Invalid card image at page {page_idx + 1}, position {card_idx + 1}")
                return None
            
            # Extract text attributes
            logger.debug(f"Extracting text from card at page {page_idx + 1}, position {card_idx + 1}")
            extracted_data = self.text_extractor.extract_card_attributes(card_image)
            
            if not extracted_data:
                logger.warning(f"No text extracted from card at page {page_idx + 1}, position {card_idx + 1}")
                return None
            
            # Attempt card identification
            identification_result = self.card_identifier.identify_card_comprehensive(
                card_image,
                extracted_data.get('name'),
                extracted_data.get('all_text'),
                extracted_data.get('set_code')
            )
            
            # Save card image
            image_filename = f"{collection_name}_p{page_idx+1:03d}_c{card_idx+1:02d}.png"
            image_path = os.path.join(settings.CARD_IMAGES_PATH, image_filename)
            cv2.imwrite(image_path, card_image)
            
            # Compile card data
            card_data = self._compile_card_data(
                extracted_data, identification_result, 
                image_path, pdf_path, page_idx, card_idx, collection_name
            )
            
            logger.debug(f"Processed card: {card_data.get('name', 'Unknown')} "
                        f"(confidence: {card_data.get('confidence_score', 0):.2f})")
            
            return card_data
            
        except Exception as e:
            logger.error(f"Individual card processing failed at page {page_idx + 1}, "
                        f"position {card_idx + 1}: {str(e)}")
            return None
    
    def _compile_card_data(self, extracted_data: Dict, identification_result: Dict,
                          image_path: str, pdf_path: str, page_idx: int, 
                          card_idx: int, collection_name: str) -> Dict:
        """Compile final card data from all sources"""
        
        # Start with extracted OCR data
        card_data = {
            'name': extracted_data.get('name', 'Unknown'),
            'mana_cost': extracted_data.get('mana_cost'),
            'type_line': extracted_data.get('type_line'),
            'oracle_text': extracted_data.get('oracle_text'),
            'flavor_text': extracted_data.get('flavor_text'),
            'rarity': extracted_data.get('rarity'),
            'colors': extracted_data.get('colors', ''),
            'converted_mana_cost': extracted_data.get('converted_mana_cost', 0),
            
            # Power/Toughness
            'power': None,
            'toughness': None,
            
            # Processing metadata
            'image_path': image_path,
            'source_pdf': pdf_path,
            'source_page': page_idx + 1,
            'card_position': card_idx + 1,
            'collection_name': collection_name,
            'confidence_score': extracted_data.get('confidence', 0.3),
            'identification_method': 'ocr_only'
        }
        
        # Add P/T if available
        pt_data = extracted_data.get('power_toughness')
        if pt_data:
            card_data['power'] = pt_data.get('power')
            card_data['toughness'] = pt_data.get('toughness')
        
        # Override with API data if identification was successful
        if identification_result.get('success') and validate_identification_result(identification_result):
            api_data = identification_result['card_data']
            
            # Update with more reliable API data
            if api_data.get('name'):
                card_data['name'] = api_data['name']
            
            if api_data.get('mana_cost'):
                card_data['mana_cost'] = api_data['mana_cost']
                card_data['converted_mana_cost'] = api_data.get('cmc', 0)
            
            if api_data.get('type_line'):
                card_data['type_line'] = api_data['type_line']
            
            if api_data.get('oracle_text'):
                card_data['oracle_text'] = api_data['oracle_text']
            
            if api_data.get('flavor_text'):
                card_data['flavor_text'] = api_data['flavor_text']
            
            if api_data.get('power'):
                card_data['power'] = api_data['power']
            
            if api_data.get('toughness'):
                card_data['toughness'] = api_data['toughness']
            
            if api_data.get('rarity'):
                card_data['rarity'] = api_data['rarity']
            
            if api_data.get('set'):
                card_data['set_code'] = api_data['set']
            
            if api_data.get('set_name'):
                card_data['set_name'] = api_data['set_name']
            
            if api_data.get('collector_number'):
                card_data['collector_number'] = api_data['collector_number']
            
            if api_data.get('colors'):
                card_data['colors'] = ''.join(api_data['colors'])
            
            if api_data.get('color_identity'):
                card_data['color_identity'] = ''.join(api_data['color_identity'])
            
            if api_data.get('id'):
                card_data['scryfall_id'] = api_data['id']
            
            # Update confidence and method
            card_data['confidence_score'] = identification_result['confidence']
            card_data['identification_method'] = identification_result['method']
        
        # Final validation
        card_data['confidence_score'] = OCRValidator.calculate_confidence_score(card_data)
        
        return card_data
    
    def _validate_card_image(self, card_image: np.ndarray) -> bool:
        """Validate that a card image is suitable for processing"""
        if card_image is None or card_image.size == 0:
            return False
        
        height, width = card_image.shape[:2]
        
        # Check minimum dimensions
        if height < 100 or width < 100:
            return False
        
        # Check aspect ratio (MTG cards should be roughly rectangular)
        aspect_ratio = width / height
        if not (0.5 < aspect_ratio < 2.0):
            return False
        
        return True
    
    def _store_cards_batch(self, cards_data: List[Dict], collection_name: str) -> Dict:
        """Store processed cards in database with duplicate detection"""
        session = self.db_manager.get_session()
        card_repo = CardRepository(session)
        
        try:
            new_cards = 0
            duplicates = 0
            errors = 0
            
            for card_data in cards_data:
                try:
                    # Skip cards without valid names
                    if not card_data.get('name') or card_data.get('name') == 'Unknown':
                        logger.warning(f"Skipping card without valid name: {card_data}")
                        errors += 1
                        continue
                    
                    # Check for duplicates
                    duplicate_result = self.duplicate_handler.detect_duplicates(
                        card_data, session
                    )
                    
                    if duplicate_result['is_duplicate']:
                        duplicates += 1
                        logger.info(f"Duplicate detected: {card_data.get('name')} "
                                  f"(confidence: {duplicate_result['confidence']:.2f})")
                        
                        # Skip marking duplicates for now to avoid database issues
                        # TODO: Fix duplicate relationship tracking properly
                        logger.debug("Skipping duplicate relationship recording to avoid database errors")
                    else:
                        # Create new card
                        card = card_repo.create_card(card_data)
                        new_cards += 1
                        logger.debug(f"Created new card: {card_data.get('name')}")
                
                except Exception as e:
                    errors += 1
                    logger.error(f"Error storing card {card_data.get('name', 'Unknown')}: {str(e)}")
                    # Rollback this card's transaction and continue
                    session.rollback()
            
            # Commit all successful changes
            try:
                session.commit()
            except Exception as e:
                logger.error(f"Failed to commit batch: {str(e)}")
                session.rollback()
                return {
                    'new_cards': 0,
                    'duplicates': 0,
                    'errors': len(cards_data)
                }
            
            return {
                'new_cards': new_cards,
                'duplicates': duplicates,
                'errors': errors
            }
            
        except Exception as e:
            session.rollback()
            logger.error(f"Batch storage failed: {str(e)}")
            return {
                'new_cards': 0,
                'duplicates': 0,
                'errors': len(cards_data)
            }
        finally:
            session.close()
    
    def _generate_exports(self, collection_name: str) -> Dict:
        """Generate export files for the processed collection"""
        try:
            exports = {}
            
            # CSV Export
            csv_result = self.export_manager.export_collection_to_csv(
                collection_name,
                os.path.join(settings.EXPORTS_PATH, f"{collection_name}.csv")
            )
            exports['csv'] = csv_result
            
            # JSON Export
            json_result = self.export_manager.export_collection_to_json(
                collection_name,
                os.path.join(settings.EXPORTS_PATH, f"{collection_name}.json")
            )
            exports['json'] = json_result
            
            # Collection Report
            report_result = self.export_manager.generate_collection_report(
                collection_name,
                os.path.join(settings.EXPORTS_PATH, f"{collection_name}_report.json")
            )
            exports['report'] = report_result
            
            logger.info(f"Generated exports for collection: {collection_name}")
            return exports
            
        except Exception as e:
            logger.error(f"Export generation failed: {str(e)}")
            return {'error': str(e)}
    
    def process_single_card_image(self, image_path: str) -> Dict:
        """Process a single card image file"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {'success': False, 'error': 'Could not load image'}
            
            # Process the card
            extracted_data = self.text_extractor.extract_card_attributes(image)
            identification_result = self.card_identifier.identify_card_comprehensive(
                image,
                extracted_data.get('name'),
                extracted_data.get('all_text')
            )
            
            return {
                'success': True,
                'extracted_data': extracted_data,
                'identification_result': identification_result,
                'image_path': image_path
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_collection_statistics(self, collection_name: str = None) -> Dict:
        """Get statistics for a collection or all collections"""
        session = self.db_manager.get_session()
        card_repo = CardRepository(session)
        
        try:
            stats = card_repo.get_collection_stats(collection_name)
            return stats
        finally:
            session.close()
    
    def search_cards(self, search_term: str, filters: Dict = None) -> List[Dict]:
        """Search for cards in the database"""
        session = self.db_manager.get_session()
        card_repo = CardRepository(session)
        
        try:
            cards = card_repo.search_cards(search_term, filters)
            return [self._card_to_dict(card) for card in cards]
        finally:
            session.close()
    
    def _card_to_dict(self, card: Card) -> Dict:
        """Convert Card model to dictionary"""
        return {
            'id': str(card.id),
            'name': card.name,
            'mana_cost': card.mana_cost,
            'converted_mana_cost': card.converted_mana_cost,
            'type_line': card.type_line,
            'oracle_text': card.oracle_text,
            'flavor_text': card.flavor_text,
            'power': card.power,
            'toughness': card.toughness,
            'set_code': card.set_code,
            'set_name': card.set_name,
            'rarity': card.rarity,
            'collector_number': card.collector_number,
            'colors': card.colors,
            'color_identity': card.color_identity,
            'collection_name': card.collection_name,
            'confidence_score': card.confidence_score,
            'identification_method': card.identification_method,
            'processed_date': card.processed_date.isoformat() if card.processed_date else None
        }

class BatchProcessor:
    """Handles batch processing of multiple PDF files"""
    
    def __init__(self, main_processor: MTGCardProcessingSystem):
        self.processor = main_processor
        self.batch_size = settings.BATCH_SIZE
    
    def process_pdf_directory(self, directory_path: str, 
                            collection_prefix: str = "batch") -> Dict:
        """Process all PDF files in a directory"""
        import glob
        
        pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
        
        if not pdf_files:
            return {
                'success': False,
                'error': 'No PDF files found in directory',
                'directory': directory_path
            }
        
        results = []
        total_cards = 0
        total_duplicates = 0
        
        for i, pdf_file in enumerate(pdf_files):
            try:
                collection_name = f"{collection_prefix}_{i+1:03d}_{os.path.basename(pdf_file).replace('.pdf', '')}"
                
                logger.info(f"Processing batch file {i+1}/{len(pdf_files)}: {pdf_file}")
                
                result = self.processor.process_pdf_collection(pdf_file, collection_name)
                results.append(result)
                
                if result.get('success'):
                    total_cards += result.get('cards_stored', 0)
                    total_duplicates += result.get('duplicates_found', 0)
                
            except Exception as e:
                logger.error(f"Batch processing failed for {pdf_file}: {str(e)}")
                results.append({
                    'success': False,
                    'pdf_file': pdf_file,
                    'error': str(e)
                })
        
        return {
            'success': True,
            'directory': directory_path,
            'files_processed': len(pdf_files),
            'total_cards_stored': total_cards,
            'total_duplicates': total_duplicates,
            'individual_results': results
        }

class ProgressTracker:
    """Tracks and reports processing progress"""
    
    def __init__(self):
        self.current_stage = "Initializing"
        self.progress_percentage = 0
        self.details = {}
        self.start_time = None
        
    def start_processing(self, total_items: int = None):
        """Start tracking a new processing job"""
        self.start_time = datetime.now()
        self.progress_percentage = 0
        self.current_stage = "Starting"
        self.details = {'total_items': total_items}
        
    def update_stage(self, stage: str, progress: float = None, details: Dict = None):
        """Update current processing stage"""
        self.current_stage = stage
        if progress is not None:
            self.progress_percentage = min(100, max(0, progress))
        if details:
            self.details.update(details)
            
    def get_status(self) -> Dict:
        """Get current processing status"""
        elapsed_time = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        return {
            'stage': self.current_stage,
            'progress_percentage': self.progress_percentage,
            'elapsed_time': elapsed_time,
            'details': self.details
        }

# Utility functions for common operations
def validate_pdf_file(pdf_path: str) -> bool:
    """Validate that a PDF file exists and is readable"""
    if not os.path.exists(pdf_path):
        return False
    
    if not pdf_path.lower().endswith('.pdf'):
        return False
    
    try:
        # Try to open with PyMuPDF to validate
        import fitz
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
        return page_count > 0
    except:
        return False

def setup_processing_directories():
    """Ensure all required directories exist"""
    directories = [
        settings.STORAGE_BASE_PATH,
        settings.CARD_IMAGES_PATH,
        settings.PDF_INPUT_PATH,
        settings.EXPORTS_PATH,
        "./logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def cleanup_old_files(days_old: int = 30):
    """Clean up old processing files"""
    import time
    
    directories_to_clean = [
        settings.CARD_IMAGES_PATH,
        settings.EXPORTS_PATH,
        "./logs"
    ]
    
    cutoff_time = time.time() - (days_old * 24 * 60 * 60)
    cleaned_count = 0
    
    for directory in directories_to_clean:
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    file_time = os.path.getmtime(file_path)
                    if file_time < cutoff_time:
                        try:
                            os.remove(file_path)
                            cleaned_count += 1
                        except OSError:
                            pass
    
    logger.info(f"Cleaned up {cleaned_count} old files")
    return cleaned_count

# Main execution function
def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MTG Card Processing System')
    parser.add_argument('pdf_path', help='Path to PDF file to process')
    parser.add_argument('--collection', help='Collection name', default=None)
    parser.add_argument('--batch', action='store_true', help='Process directory of PDFs')
    parser.add_argument('--cleanup', type=int, help='Clean up files older than N days', default=None)
    
    args = parser.parse_args()
    
    # Setup directories
    setup_processing_directories()
    
    # Handle cleanup if requested
    if args.cleanup:
        cleanup_old_files(args.cleanup)
        return
    
    # Initialize processing system
    try:
        processor = MTGCardProcessingSystem()
        
        if args.batch:
            # Batch processing mode
            batch_processor = BatchProcessor(processor)
            result = batch_processor.process_pdf_directory(
                args.pdf_path, 
                args.collection or "batch"
            )
        else:
            # Single file processing mode
            if not validate_pdf_file(args.pdf_path):
                print(f"Error: Invalid PDF file: {args.pdf_path}")
                return
            
            result = processor.process_pdf_collection(
                args.pdf_path, 
                args.collection
            )
        
        # Print results
        print("\n" + "="*50)
        print("PROCESSING RESULTS")
        print("="*50)
        print(json.dumps(result, indent=2, default=str))
        
        if result.get('success'):
            print(f"\n✓ Processing completed successfully!")
            if not args.batch:
                print(f"✓ Cards stored: {result.get('cards_stored', 0)}")
                print(f"✓ Duplicates found: {result.get('duplicates_found', 0)}")
                print(f"✓ Processing time: {result.get('processing_time', 0):.2f} seconds")
        else:
            print(f"\n✗ Processing failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        logger.exception("Fatal error during processing")

if __name__ == "__main__":
    main()