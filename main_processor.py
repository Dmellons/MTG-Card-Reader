"""
Main Processing System - Orchestrates the complete MTG card processing workflow
"""
import os
import logging
import json
import pandas as pd
import re
import uuid
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
            
            # Store the card even if name extraction failed (for debugging and improvement)
            # We'll create placeholder names for cards without extracted names or with suspicious names
            extracted_name = extracted_data.get('name', '')
            
            # Check for various suspicious patterns that indicate OCR errors
            def has_ocr_corruption(name: str) -> bool:
                """Check if a name shows signs of OCR corruption"""
                if not name:
                    return False
                name_lower = name.lower()
                
                # Common OCR corruption patterns
                corruption_patterns = [
                    'cgunt',      # "Cguntergpell" (Counterspell)
                    'spelljack',  # False matches
                    'gcout',      # "Gcout" (Scout)
                    'gp',         # "gp" substitutions
                    'cg',         # "cg" substitutions at start
                ]
                
                return any(pattern in name_lower for pattern in corruption_patterns)
            
            is_suspicious_name = (
                not extracted_name or  # No name
                len(extracted_name) <= 4 or  # Very short name (like "jng", "Scuc")
                not any(c.isalpha() for c in extracted_name) or  # No letters
                extracted_name.lower() in ['page', 'card', 'unknown', 'null', 'none'] or  # Common placeholders
                (extracted_name.lower().endswith('unculus') and 'hazy' in extracted_data.get('all_text', '').lower()) or  # "hornunculus" should be "Hazy Homunculus"
                extracted_name.lower() in ['hornunculus', 'homunculus'] or  # Common OCR errors for card names
                has_ocr_corruption(extracted_name)  # Names with OCR corruption patterns
            )
            
            if is_suspicious_name:
                logger.info(f"Suspicious name detected: '{extracted_name}' - attempting enhanced identification")
                # Try to find a better name from the extracted text
                better_name = self._extract_name_from_text(extracted_data.get('all_text', ''))
                
                # Also try pattern matching (even if we got a name from text extraction, 
                # pattern matching might be more accurate for cards with OCR errors)
                pattern_name = self._identify_card_from_full_text(extracted_data.get('all_text', ''))
                
                # Prefer pattern matching result if available, otherwise use text extraction
                if pattern_name:
                    extracted_data['name'] = pattern_name
                    logger.debug(f"Used pattern matching result: {pattern_name}")
                elif better_name:
                    extracted_data['name'] = better_name
                    logger.debug(f"Used text extraction result: {better_name}")
                else:
                    # Create a placeholder name based on position and any extracted text
                    page_card_id = f"Page_{page_idx+1}_Card_{card_idx+1}"
                    if extracted_data.get('all_text'):
                        # Use first few words of extracted text as hint
                        text_hint = ' '.join(extracted_data['all_text'].split()[:3])
                        if text_hint and len(text_hint) > 5:  # Only use if meaningful
                            extracted_data['name'] = f"{page_card_id}_{text_hint}"
                        else:
                            extracted_data['name'] = page_card_id
                    else:
                        extracted_data['name'] = page_card_id
                    logger.debug(f"Created placeholder name: {extracted_data['name']}")
            
            # Only skip if we have absolutely no data at all
            if (not extracted_data.get('all_text') and 
                extracted_data.get('confidence', 0) < 0.05):
                logger.warning(f"Absolutely no data extracted from card at page {page_idx + 1}, position {card_idx + 1}")
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
            
            # Add processing metadata for debugging (but don't include in database storage)
            processing_metadata = {
                'ocr_confidence': extracted_data.get('confidence', 0),
                'identification_confidence': identification_result.get('confidence', 0) if identification_result else 0,
                'has_extracted_name': bool(extracted_data.get('name')),
                'extracted_text_length': len(extracted_data.get('all_text', '')),
                'is_placeholder_name': card_data.get('name', '').startswith('Page_'),
            }
            
            logger.info(f"Processed card: {card_data.get('name', 'Unknown')} "
                       f"(OCR conf: {extracted_data.get('confidence', 0):.2f}, "
                       f"ID conf: {card_data.get('confidence_score', 0):.2f}, "
                       f"Text len: {len(extracted_data.get('all_text', ''))})")
            
            return card_data
            
        except Exception as e:
            error_msg = str(e)
            if "'NoneType' object has no attribute 'lower'" in error_msg:
                # Add more debug info for this specific error
                logger.error(f"NoneType.lower() error at page {page_idx + 1}, position {card_idx + 1}. "
                           f"Extracted data: {extracted_data.get('type_line') if 'extracted_data' in locals() else 'N/A'}, "
                           f"Identification result: {identification_result.get('card_data', {}).get('type_line') if 'identification_result' in locals() and identification_result else 'N/A'}")
            logger.error(f"Individual card processing failed at page {page_idx + 1}, "
                        f"position {card_idx + 1}: {error_msg}")
            return None
    
    def _compile_card_data(self, extracted_data: Dict, identification_result: Dict,
                          image_path: str, pdf_path: str, page_idx: int, 
                          card_idx: int, collection_name: str) -> Dict:
        """Compile final card data from all sources"""
        
        # Start with extracted OCR data
        card_data = {
            'name': extracted_data.get('name') or None,  # Keep None instead of 'Unknown'
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
                # Convert string UUID to UUID object for database compatibility
                try:
                    card_data['scryfall_id'] = uuid.UUID(api_data['id'])
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid Scryfall ID format: {api_data['id']}, error: {e}")
                    # Don't store invalid UUIDs
            
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
                    # Only skip cards if they have absolutely no name (shouldn't happen now)
                    if not card_data.get('name'):
                        logger.warning(f"Skipping card without any name: {card_data}")
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
                        # Create new card - filter to only valid database fields
                        db_card_data = self._filter_card_data_for_db(card_data)
                        card = card_repo.create_card(db_card_data)
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
    
    def _filter_card_data_for_db(self, card_data: Dict) -> Dict:
        """Filter card data to only include valid database fields"""
        # Valid fields from Card model
        valid_fields = {
            'name', 'scryfall_id', 'mana_cost', 'converted_mana_cost', 'type_line',
            'oracle_text', 'flavor_text', 'power', 'toughness', 'set_code', 'set_name',
            'rarity', 'collector_number', 'colors', 'color_identity', 'image_path',
            'source_pdf', 'source_page', 'card_position', 'processed_date',
            'confidence_score', 'identification_method', 'collection_name'
        }
        
        # Filter to only valid fields
        filtered_data = {
            key: value for key, value in card_data.items() 
            if key in valid_fields and value is not None
        }
        
        return filtered_data
    
    def _extract_name_from_text(self, all_text: str) -> Optional[str]:
        """Try to extract a reasonable card name from the full OCR text"""
        if not all_text:
            return None
        
        # Clean up the text
        text = all_text.strip()
        words = text.split()
        
        if not words:
            return None
        
        # Look for words that could be card names (avoiding common non-name words)
        non_name_words = {
            'creature', 'instant', 'sorcery', 'enchantment', 'artifact', 'land', 'planeswalker',
            'flying', 'trample', 'first', 'strike', 'haste', 'vigilance', 'reach', 'deathtouch',
            'when', 'whenever', 'until', 'end', 'turn', 'target', 'player', 'damage', 'draw', 'card',
            'mana', 'tap', 'untap', 'counter', 'spell', 'ability', 'cost', 'pay', 'sacrifice',
            'destroy', 'exile', 'graveyard', 'hand', 'library', 'battlefield', 'play',
            'illus', 'illustration', 'art', 'artist', 'copyright', 'wizards', 'coast', 'inc',
            'unblockable', 'blocked', 'blocking', 'enters', 'leaves', 'comes', 'into', 'from',
            'defending', 'attacking', 'controls', 'controlled', 'owner', 'owners', 'control'
        }
        
        # Try to find a good candidate name from the first few words
        for i in range(min(4, len(words))):  # Check first 4 words
            word = words[i].strip('.,!?:;')
            
            # Skip if it's a common non-name word, number, or very short
            if (len(word) >= 3 and 
                word.lower() not in non_name_words and
                not word.isdigit() and
                not re.match(r'^[\d\{\}/\*\(\)]+$', word)):
                
                # If it's a reasonable length, check for multi-word names first
                if 3 <= len(word) <= 25:
                    # If it's part of a multi-word name, try to get 2-3 words
                    if i < len(words) - 1:
                        next_word = words[i + 1].strip('.,!?:;')
                        if (len(next_word) >= 3 and 
                            next_word.lower() not in non_name_words and
                            not next_word.isdigit()):
                            
                            # Check if there's a third word too
                            if i < len(words) - 2:
                                third_word = words[i + 2].strip('.,!?:;')
                                if (len(third_word) >= 3 and 
                                    third_word.lower() not in non_name_words and
                                    not third_word.isdigit() and
                                    len(f"{word} {next_word} {third_word}") <= 30):
                                    return f"{word} {next_word} {third_word}"
                            
                            # Use two words if reasonable length
                            if len(f"{word} {next_word}") <= 25:
                                return f"{word} {next_word}"
                    
                    # Fall back to single word if multi-word didn't work
                    return word
        
        return None
    
    def _identify_card_from_full_text(self, all_text: str) -> Optional[str]:
        """Try to identify the actual card by searching for key phrases in the text"""
        if not all_text or len(all_text) < 10:
            return None
        
        # Import here to avoid circular imports
        from card_identifier import MTGCardIdentifier
        
        try:
            text = all_text.lower()
            words = all_text.split()
            
            # Strategy 1: Look for unique ability text that might match cards
            # Common MTG ability patterns that are unique to specific cards
            unique_patterns = [
                ('hazy homunculus', 'unblockable.*defending player controls.*untapped land'),
                ('thalakos scout', 'shadow.*choose and discard.*return.*scout.*owner'),
                ('wormfang crab', 'unblockable.*opponent chooses.*permanent.*removes it from.*game'),
                ('spiketail drake', 'flying.*sacrifice.*spiketail.*counter target.*unless.*controller pays'),
                ('phantom warrior', 'phantom warrior.*unblockable'),
                ('capsize', 'buyback.*return target permanent.*owner.*hand'),
                ('counterspell', 'counter target spell'),
                ('heightened awareness', 'discard your hand.*beginning.*draw step.*draw.*card')
            ]
            
            identifier = MTGCardIdentifier()
            
            # Try to match unique patterns
            for card_name, pattern in unique_patterns:
                if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                    logger.info(f"PATTERN MATCH found for {card_name}: {pattern[:50]}...")
                    # Try to identify this card
                    result = identifier.identify_by_name(card_name)
                    if result:
                        logger.info(f"Successfully identified card: {card_name} -> {result.get('name')}")
                        return result.get('name')
                    else:
                        logger.warning(f"Pattern matched but API lookup failed for: {card_name}")
            
            # Strategy 2: Try common combinations of words that might be card names
            for i in range(len(words) - 1):
                for j in range(i + 1, min(i + 4, len(words))):  # Try 2-3 word combinations
                    candidate = ' '.join(words[i:j+1])
                    candidate = re.sub(r'[^\w\s]', '', candidate)  # Remove punctuation
                    
                    if (len(candidate) >= 6 and 
                        len(candidate) <= 30 and
                        not candidate.lower().startswith(('when', 'whenever', 'target', 'return', 'draw', 'counter'))):
                        
                        # Try fuzzy matching with this candidate
                        result = identifier.identify_by_name(candidate)
                        if result:
                            logger.debug(f"Found card via fuzzy matching: {candidate} -> {result.get('name')}")
                            return result.get('name')
            
            return None
            
        except Exception as e:
            logger.debug(f"Card identification from full text failed: {str(e)}")
            return None
    
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