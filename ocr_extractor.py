"""
OCR and Text Extraction Module for MTG Cards
Supports multiple OCR engines with MTG-specific text parsing
"""
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import re
import logging
from typing import Dict, List, Optional, Tuple
from config import settings

# Import OCR libraries with fallback handling
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logging.warning("EasyOCR not available")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.debug("Tesseract not available - using EasyOCR only")

logger = logging.getLogger(__name__)

class MTGTextExtractor:
    """Main text extraction class for MTG cards"""
    
    def __init__(self, ocr_engine: str = None, use_cloud_ocr: bool = False):
        self.ocr_engine = ocr_engine or settings.OCR_ENGINE
        self.use_cloud_ocr = use_cloud_ocr or settings.USE_CLOUD_OCR
        self.confidence_threshold = settings.OCR_CONFIDENCE_THRESHOLD
        
        # Initialize OCR readers
        self.easyocr_reader = None
        if EASYOCR_AVAILABLE and self.ocr_engine in ['easyocr', 'both']:
            try:
                import warnings
                with warnings.catch_warnings():
                    # Suppress PyTorch pin_memory warning when no GPU is available
                    warnings.filterwarnings("ignore", message=".*pin_memory.*")
                    
                    # Try to use GPU if available, fallback to CPU
                    try:
                        import torch
                        gpu_available = torch.cuda.is_available()
                        logger.info(f"CUDA available: {gpu_available}")
                        if gpu_available:
                            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
                        self.easyocr_reader = easyocr.Reader(['en'], gpu=gpu_available, verbose=False)
                    except Exception as gpu_error:
                        logger.warning(f"GPU initialization failed, using CPU: {gpu_error}")
                        self.easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                logger.info("EasyOCR initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {str(e)}")
        
        # MTG-specific regex patterns
        self.mana_pattern = r'\{[WUBRGCTXYS0-9]+(?:/[WUBRG])?\}'
        self.pt_pattern = r'(\d+|\*)\/(\d+|\*)'
        self.type_patterns = {
            'creature': r'\b(?:Creature|creature)\b',
            'instant': r'\b(?:Instant|instant)\b',
            'sorcery': r'\b(?:Sorcery|sorcery)\b',
            'enchantment': r'\b(?:Enchantment|enchantment)\b',
            'artifact': r'\b(?:Artifact|artifact)\b',
            'planeswalker': r'\b(?:Planeswalker|planeswalker)\b',
            'land': r'\b(?:Land|land)\b'
        }
        
        # Template matching for card regions
        self.card_templates = self._load_card_templates()
        
        # Common MTG keywords for validation
        self.mtg_keywords = {
            'flying', 'trample', 'haste', 'vigilance', 'deathtouch', 'lifelink',
            'first strike', 'double strike', 'hexproof', 'shroud', 'defender',
            'reach', 'flash', 'prowess', 'menace', 'indestructible'
        }
    
    def _load_card_templates(self) -> Dict:
        """Load card region templates from sample cards"""
        templates = {}
        
        try:
            import os
            sample_cards_path = "./data/sample_cards/"
            
            if os.path.exists(sample_cards_path):
                for filename in os.listdir(sample_cards_path):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        template_path = os.path.join(sample_cards_path, filename)
                        template_image = cv2.imread(template_path)
                        
                        if template_image is not None:
                            # Extract template regions
                            template_name = os.path.splitext(filename)[0]
                            templates[template_name] = self._extract_template_regions(template_image)
                            
                logger.info(f"Loaded {len(templates)} card templates")
                            
        except Exception as e:
            logger.warning(f"Failed to load card templates: {str(e)}")
            
        return templates
    
    def _enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """Enhance image quality for better OCR on poor quality scans"""
        try:
            # Apply multiple enhancement strategies
            enhanced_versions = []
            
            # Strategy 1: Bilateral filter to reduce noise while preserving edges
            bilateral = cv2.bilateralFilter(image, 9, 75, 75)
            enhanced_versions.append(bilateral)
            
            # Strategy 2: Morphological operations to clean up text
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Erosion followed by dilation to remove noise
            kernel = np.ones((2,2), np.uint8)
            eroded = cv2.erode(gray, kernel, iterations=1)
            dilated = cv2.dilate(eroded, kernel, iterations=1)
            
            # Convert back to original format
            if len(image.shape) == 3:
                morph_enhanced = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
            else:
                morph_enhanced = dilated
            enhanced_versions.append(morph_enhanced)
            
            # Strategy 3: Adaptive threshold for better text contrast
            adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                  cv2.THRESH_BINARY, 11, 2)
            if len(image.shape) == 3:
                adaptive_enhanced = cv2.cvtColor(adaptive_thresh, cv2.COLOR_GRAY2BGR)
            else:
                adaptive_enhanced = adaptive_thresh
            enhanced_versions.append(adaptive_enhanced)
            
            # Strategy 4: Original approach with improved parameters
            blurred = cv2.GaussianBlur(image, (3, 3), 0)
            sharpening_kernel = np.array([[-1,-1,-1],
                                        [-1, 9,-1],
                                        [-1,-1,-1]])
            sharpened = cv2.filter2D(blurred, -1, sharpening_kernel)
            
            if len(image.shape) == 3:
                lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                lab[:,:,0] = clahe.apply(lab[:,:,0])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                enhanced = clahe.apply(sharpened)
            enhanced_versions.append(enhanced)
            
            # Return the bilateral filtered version as it tends to work best for text
            return enhanced_versions[0]
            
        except Exception as e:
            logger.debug(f"Image enhancement failed: {str(e)}")
            return image
    
    def _extract_template_regions(self, template_image: np.ndarray) -> Dict:
        """Extract key regions from a template card image"""
        height, width = template_image.shape[:2]
        
        regions = {
            'name_region': template_image[0:int(height * 0.12), :],
            'mana_cost_region': template_image[0:int(height * 0.12), int(width * 0.7):],
            'type_line_region': template_image[int(height * 0.55):int(height * 0.65), :],
            'text_box_region': template_image[int(height * 0.4):int(height * 0.85), :],
            'pt_region': template_image[int(height * 0.85):, int(width * 0.7):],
        }
        
        return regions
    
    def enhance_extraction_with_templates(self, card_image: np.ndarray, extracted_data: Dict) -> Dict:
        """Enhance extraction results using template matching"""
        if not self.card_templates:
            return extracted_data
        
        try:
            # Find best matching template
            best_template = self._find_best_template_match(card_image)
            
            if best_template:
                # Use template guidance to improve extraction
                enhanced_data = self._apply_template_guidance(card_image, extracted_data, best_template)
                return enhanced_data
                
        except Exception as e:
            logger.warning(f"Template-based enhancement failed: {str(e)}")
        
        return extracted_data
    
    def _find_best_template_match(self, card_image: np.ndarray) -> Optional[str]:
        """Find the best matching template for the card"""
        if not self.card_templates:
            return None
        
        best_match = None
        best_score = 0
        
        # Convert card to grayscale for comparison
        if len(card_image.shape) == 3:
            card_gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
        else:
            card_gray = card_image
        
        # Resize card to standard size for comparison
        standard_height = 600
        aspect_ratio = card_image.shape[1] / card_image.shape[0]
        standard_width = int(standard_height * aspect_ratio)
        card_resized = cv2.resize(card_gray, (standard_width, standard_height))
        
        for template_name, template_regions in self.card_templates.items():
            try:
                # Compare overall structure
                score = self._calculate_template_similarity(card_resized, template_regions)
                
                if score > best_score:
                    best_score = score
                    best_match = template_name
                    
            except Exception as e:
                logger.debug(f"Template comparison failed for {template_name}: {str(e)}")
        
        if best_score > 0.3:  # Minimum similarity threshold
            logger.debug(f"Best template match: {best_match} (score: {best_score:.2f})")
            return best_match
        
        return None
    
    def _calculate_template_similarity(self, card_image: np.ndarray, template_regions: Dict) -> float:
        """Calculate similarity between card and template"""
        try:
            # Simple structural similarity based on edge detection
            card_edges = cv2.Canny(card_image, 50, 150)
            
            # Compare key regions if template regions are available
            similarity_scores = []
            
            for region_name, region_template in template_regions.items():
                if region_template is not None and region_template.size > 0:
                    # Convert template region to same format
                    if len(region_template.shape) == 3:
                        template_gray = cv2.cvtColor(region_template, cv2.COLOR_BGR2GRAY)
                    else:
                        template_gray = region_template
                    
                    # Resize template to match card proportions
                    template_resized = cv2.resize(template_gray, (card_image.shape[1], card_image.shape[0]))
                    template_edges = cv2.Canny(template_resized, 50, 150)
                    
                    # Calculate normalized cross-correlation
                    correlation = cv2.matchTemplate(card_edges, template_edges, cv2.TM_CCOEFF_NORMED)
                    max_val = correlation.max()
                    similarity_scores.append(max_val)
            
            if similarity_scores:
                return sum(similarity_scores) / len(similarity_scores)
            
        except Exception as e:
            logger.debug(f"Template similarity calculation failed: {str(e)}")
        
        return 0.0
    
    def _apply_template_guidance(self, card_image: np.ndarray, extracted_data: Dict, template_name: str) -> Dict:
        """Apply template guidance to improve extraction accuracy"""
        enhanced_data = extracted_data.copy()
        
        try:
            # Use template knowledge to guide region extraction
            if template_name in self.card_templates:
                template_regions = self.card_templates[template_name]
                
                # If card name extraction failed, try template-guided extraction
                if not enhanced_data.get('name') or len(enhanced_data.get('name', '')) < 3:
                    template_name_result = self._extract_with_template_guidance(
                        card_image, 'name_region', template_regions
                    )
                    if template_name_result:
                        enhanced_data['name'] = template_name_result
                        
                # Apply to other regions as needed
                # Similar logic can be applied for other card attributes
                
        except Exception as e:
            logger.warning(f"Template guidance application failed: {str(e)}")
        
        return enhanced_data
    
    def _extract_with_template_guidance(self, card_image: np.ndarray, region_name: str, template_regions: Dict) -> Optional[str]:
        """Extract text from a specific region using template guidance"""
        try:
            height, width = card_image.shape[:2]
            
            # Define region coordinates based on template knowledge
            region_coords = {
                'name_region': (0, int(height * 0.12), 0, width),
                'type_line_region': (int(height * 0.55), int(height * 0.65), 0, width),
                'text_box_region': (int(height * 0.4), int(height * 0.85), 0, width),
            }
            
            if region_name in region_coords:
                y1, y2, x1, x2 = region_coords[region_name]
                region_image = card_image[y1:y2, x1:x2]
                
                # Apply enhanced preprocessing for this region
                processed_region = self._preprocess_for_ocr(region_image)
                
                # Perform OCR on the region
                ocr_results = self._perform_ocr(processed_region)
                
                if ocr_results:
                    # Extract best text from results
                    best_text = None
                    best_confidence = 0
                    
                    for result in ocr_results:
                        text = result.get('text', '').strip()
                        confidence = result.get('confidence', 0)
                        
                        if confidence > best_confidence and len(text) > 2:
                            best_text = text
                            best_confidence = confidence
                    
                    return best_text
                    
        except Exception as e:
            logger.debug(f"Template-guided extraction failed for {region_name}: {str(e)}")
        
        return None
    
    def extract_card_attributes(self, card_image: np.ndarray) -> Dict:
        """
        Extract all MTG card attributes from image
        Returns dictionary with parsed card data
        """
        try:
            # Preprocess image for better OCR
            preprocessed = self._preprocess_for_ocr(card_image)
            
            # Get OCR results with bounding boxes
            ocr_results = self._perform_ocr(preprocessed)
            
            if not ocr_results:
                logger.warning("No OCR results obtained")
                return {
                    'name': None,
                    'mana_cost': None,
                    'type_line': None,
                    'oracle_text': None,
                    'flavor_text': None,
                    'power_toughness': None,
                    'rarity': None,
                    'all_text': '',
                    'confidence': 0.0
                }
            
            # Try targeted extraction first for better results
            card_name = self._extract_card_name_targeted(card_image)
            
            # If targeted extraction fails, fall back to position-based extraction
            if not card_name or (isinstance(card_name, str) and len(card_name) < 3):
                card_name = self._extract_card_name(ocr_results, card_image.shape)
                logger.debug("Using position-based card name extraction as fallback")
            
            # Ensure card_name is a valid string or None
            if card_name and not isinstance(card_name, str):
                card_name = str(card_name).strip() if card_name else None
            
            card_data = {
                'name': card_name,
                'mana_cost': self._extract_mana_cost(ocr_results),
                'type_line': self._extract_type_line(ocr_results, card_image.shape),
                'oracle_text': self._extract_oracle_text(ocr_results, card_image.shape),
                'flavor_text': self._extract_flavor_text(ocr_results, card_image.shape),
                'power_toughness': self._extract_power_toughness(ocr_results, card_image.shape),
                'rarity': self._extract_rarity(ocr_results),
                'all_text': ' '.join([item['text'] for item in ocr_results]),
                'confidence': self._calculate_overall_confidence(ocr_results)
            }
            
            # Post-process and validate
            card_data = self._post_process_card_data(card_data)
            
            # Apply template-based enhancement
            card_data = self.enhance_extraction_with_templates(card_image, card_data)
            
            logger.debug(f"Extracted card data: {card_data['name']} - Confidence: {card_data['confidence']:.2f}")
            return card_data
            
        except Exception as e:
            logger.error(f"Error extracting card attributes: {str(e)}")
            return {}
    
    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Optimize image for OCR accuracy with enhanced preprocessing for white/light cards"""
        try:
            # Resize image to optimal size for OCR (aim for 300-400 DPI equivalent)
            # Increased target size for better text recognition of small or poor quality text
            height, width = image.shape[:2]
            target_height = 1000  # Increased from 800 for better OCR on poor quality scans
            if height < target_height:
                scale_factor = target_height / height
                new_width = int(width * scale_factor)
                new_height = target_height
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            
            # Additional sharpening for poor quality scans
            image = self._enhance_image_quality(image)
            
            # Convert to grayscale for processing
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Detect if this is a light/white card and adjust preprocessing accordingly
            is_light_card = self._is_light_colored_card(gray)
            
            if is_light_card:
                # Special preprocessing for white/light cards
                result = self._preprocess_light_card(gray)
                logger.debug("Applied light card preprocessing")
            else:
                # Standard preprocessing for darker cards
                result = self._preprocess_standard_card(gray)
                logger.debug("Applied standard card preprocessing")
            
            # Convert back to color if original was color
            if len(image.shape) == 3:
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            
            return result
            
        except Exception as e:
            logger.warning(f"Enhanced image preprocessing failed: {str(e)}")
            return image
    
    def _is_light_colored_card(self, gray_image: np.ndarray) -> bool:
        """Detect if card is predominantly light colored (white/light cards)"""
        try:
            # Calculate average brightness
            mean_brightness = np.mean(gray_image)
            
            # Calculate percentage of pixels above certain brightness threshold
            bright_pixels = np.sum(gray_image > 200)
            total_pixels = gray_image.size
            bright_percentage = bright_pixels / total_pixels
            
            # Consider it a light card if mean brightness is high or lots of bright pixels
            is_light = mean_brightness > 180 or bright_percentage > 0.6
            
            logger.debug(f"Card brightness analysis: mean={mean_brightness:.1f}, bright_pixels={bright_percentage:.2%}, is_light={is_light}")
            return is_light
            
        except Exception as e:
            logger.warning(f"Light card detection failed: {str(e)}")
            return False
    
    def _preprocess_light_card(self, gray_image: np.ndarray) -> np.ndarray:
        """Special preprocessing for white/light colored cards"""
        try:
            # Apply gentle denoising first
            denoised = cv2.fastNlMeansDenoising(gray_image, None, 10, 7, 21)
            
            # Invert the image temporarily to make text darker
            inverted = 255 - denoised
            
            # Apply CLAHE to the inverted image for better contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced_inverted = clahe.apply(inverted)
            
            # Invert back
            enhanced = 255 - enhanced_inverted
            
            # Apply adaptive thresholding for better text separation
            adaptive_thresh = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply morphological operations to clean up text
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
            
        except Exception as e:
            logger.warning(f"Light card preprocessing failed: {str(e)}")
            return gray_image
    
    def _preprocess_standard_card(self, gray_image: np.ndarray) -> np.ndarray:
        """Standard preprocessing for darker cards"""
        try:
            # Apply bilateral filter to reduce noise while preserving edges
            filtered = cv2.bilateralFilter(gray_image, 9, 75, 75)
            
            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(filtered)
            
            # Apply morphological operations to clean up text
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            morphed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
            
            # Apply Gaussian blur to reduce artifacts
            blurred = cv2.GaussianBlur(morphed, (1, 1), 0)
            
            return blurred
            
        except Exception as e:
            logger.warning(f"Standard card preprocessing failed: {str(e)}")
            return gray_image
    
    def _alternative_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Alternative preprocessing approach for difficult images"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Try morphological operations to clean up text
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            morphed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            # Apply Gaussian blur to smooth
            blurred = cv2.GaussianBlur(morphed, (3, 3), 0)
            
            # Apply binary threshold
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convert back to BGR if needed
            if len(image.shape) == 3:
                result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            else:
                result = binary
            
            return result
            
        except Exception as e:
            logger.warning(f"Alternative preprocessing failed: {str(e)}")
            return image
    
    def _perform_ocr(self, image: np.ndarray) -> List[Dict]:
        """Perform OCR with single best strategy and text corrections"""
        results = []
        
        try:
            # Use enhanced preprocessing for better quality
            enhanced_image = self._enhance_image_quality(image)
            results = self._try_ocr_extraction(enhanced_image)
            
            # If no results, try original image
            if not results:
                logger.debug("Enhanced OCR failed, trying original image")
                results = self._try_ocr_extraction(image)
            
            # If still no results, try with lower confidence threshold
            if not results:
                logger.debug("Retrying OCR with lower confidence threshold")
                original_threshold = self.confidence_threshold
                self.confidence_threshold = max(0.1, self.confidence_threshold - 0.1)
                
                results = self._try_ocr_extraction(enhanced_image)
                if not results:
                    results = self._try_ocr_extraction(image)
                
                self.confidence_threshold = original_threshold
            
            # Apply text corrections to improve accuracy
            if results:
                corrected_results = self._apply_text_corrections(results)
                return corrected_results
            
            return results
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {str(e)}")
            return []
    
    def _try_ocr_extraction(self, image: np.ndarray) -> List[Dict]:
        """Try OCR extraction with current engine settings"""
        results = []
        
        try:
            if self.ocr_engine == 'easyocr' and self.easyocr_reader:
                results = self._easyocr_extract(image)
            elif self.ocr_engine == 'tesseract' and TESSERACT_AVAILABLE:
                results = self._tesseract_extract(image)
            elif self.ocr_engine == 'both':
                if self.easyocr_reader:
                    results = self._easyocr_extract(image)
                if not results and TESSERACT_AVAILABLE:
                    results = self._tesseract_extract(image)
        
        except Exception as e:
            logger.debug(f"OCR extraction attempt failed: {str(e)}")
        
        return results
    
    def _create_high_contrast_binary(self, image: np.ndarray) -> np.ndarray:
        """Create high contrast binary image for difficult OCR cases"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply adaptive threshold for better text separation
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                         cv2.THRESH_BINARY, 15, 8)
            
            # Clean up with morphological operations
            kernel = np.ones((2,2), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            if len(image.shape) == 3:
                return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
            return cleaned
            
        except Exception as e:
            logger.debug(f"Binary image creation failed: {str(e)}")
            return image
    
    def _merge_ocr_results(self, all_results: List[Dict]) -> List[Dict]:
        """Merge OCR results from multiple strategies, keeping best confidence"""
        if not all_results:
            return []
        
        # Results are already flattened
        flattened = all_results
        
        if not flattened:
            return []
        
        # Group by spatial proximity and keep best confidence
        merged = {}
        threshold = 50  # pixels
        
        for result in flattened:
            bbox = result['bbox']
            center_x = sum(point[0] for point in bbox) / 4
            center_y = sum(point[1] for point in bbox) / 4
            key = (int(center_x // threshold), int(center_y // threshold))
            
            if key not in merged or result['confidence'] > merged[key]['confidence']:
                merged[key] = result
        
        return list(merged.values())
    
    def _apply_text_corrections(self, results: List[Dict]) -> List[Dict]:
        """Apply common OCR error corrections for MTG cards"""
        corrected_results = []
        
        for result in results:
            text = result['text']
            original_text = text
            
            # Common OCR character substitutions
            corrections = {
                'Cguntergpell': 'Counterspell',
                'Ceuncerepel': 'Counterspell',
                'Cbuncerepei': 'Counterspell',
                'Luightened': 'Heightened',
                'Eightened': 'Heightened',
                'haohcened': 'Heightened',
                'Awarenesz': 'Awareness',
                'Awarciese': 'Awareness',
                'hhigh': 'High',
                'huHioh': 'High',
                'igi': 'High',
                'Ixstant': 'Instant',
                'Instarut': 'Instant',
                'Enchantmcat': 'Enchantment',
                'Eachartmcate': 'Enchantment',
                'Dery': 'Very',
                'udes': 'tides',
                'ury': 'very',
                'Wl': 'We',
                'Wme': 'Come',
                'Wen': 'When',
                'reurement': 'retirement',
                'Drelv': 'Delve',
                'Khurzog': 'Herzog',
                'Herzoo': 'Herzog',
                'Ilus': 'Illus',
                'Ilius': 'Illus',
                'IMus': 'Illus',
                '61J50': '6150',
                'Hannibal': 'Hannibal',
                'Hannbal': 'Hannibal',
                'Huntbal': 'Hannibal',
                'taroet': 'target',
                'Jutnnl': 'Journal',
                'Mae': 'Mana',
                'edoe': 'edge',
                'tizard': 'wizard',
                'vizard': 'wizard',
                'alire': 'alive',
                'thc': 'the',
                'JOU': 'you',
                'Kino': 'King',
                'Turcle': 'Turtle',
                'Crealur': 'Creature',
                'Flyino': 'Flying',
                'Spikctail': 'Spiketail',
                't0': 'to',
                'L0': 'to',
                'pointino': 'pointing',
                'Tuicidel': 'Twisted',
                'Suain': 'Swain',
            }
            
            # Apply direct corrections
            for error, correction in corrections.items():
                if error in text:
                    text = text.replace(error, correction)
            
            # Character-level corrections
            char_corrections = {
                'C': 'G',  # Common OCR error: C confused with G
                'g': 'o',  # g confused with o
                'L': 'H',  # L confused with H
                'z': 's',  # z confused with s
                '1': 'l',  # 1 confused with l
                '0': 'o',  # 0 confused with o
            }
            
            # Apply character corrections only for likely card names
            if len(text) > 2 and len(text) < 30:  # Likely card name length
                for wrong, right in char_corrections.items():
                    # Only apply if it makes the text more readable
                    if wrong in text and self._would_improve_text(text, wrong, right):
                        text = text.replace(wrong, right)
            
            # Create corrected result
            corrected_result = result.copy()
            corrected_result['text'] = text
            corrected_result['original_text'] = original_text
            corrected_results.append(corrected_result)
        
        return corrected_results
    
    def _would_improve_text(self, text: str, wrong_char: str, right_char: str) -> bool:
        """Determine if character replacement would improve readability"""
        # Simple heuristic: replacement is good if it creates more vowels
        # or common letter combinations
        test_text = text.replace(wrong_char, right_char)
        
        # Count vowels
        vowels = 'aeiouAEIOU'
        original_vowels = sum(1 for c in text if c in vowels)
        test_vowels = sum(1 for c in test_text if c in vowels)
        
        # Prefer more vowels in reasonable proportions
        if test_vowels > original_vowels and test_vowels <= len(test_text) * 0.5:
            return True
        
        # Check for common English patterns
        common_patterns = ['th', 'he', 'in', 'er', 'an', 're', 'ed', 'nd', 'ha', 'et']
        original_patterns = sum(1 for pattern in common_patterns if pattern in text.lower())
        test_patterns = sum(1 for pattern in common_patterns if pattern in test_text.lower())
        
        return test_patterns > original_patterns
    
    def _easyocr_extract(self, image: np.ndarray) -> List[Dict]:
        """Extract text using EasyOCR"""
        try:
            results = self.easyocr_reader.readtext(image)
            
            formatted_results = []
            for bbox, text, confidence in results:
                if confidence >= self.confidence_threshold:
                    formatted_results.append({
                        'bbox': bbox,
                        'text': text.strip(),
                        'confidence': confidence
                    })
            
            logger.debug(f"EasyOCR extracted {len(formatted_results)} text regions")
            return formatted_results
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {str(e)}")
            return []
    
    def _tesseract_extract(self, image: np.ndarray) -> List[Dict]:
        """Extract text using Tesseract OCR"""
        try:
            # Configure Tesseract for better MTG card recognition
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789{}/*-+,.:;!?()[]'
            
            # Get detailed data with bounding boxes
            data = pytesseract.image_to_data(image, config=custom_config, output_type='dict')
            
            formatted_results = []
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                text = data['text'][i].strip()
                confidence = float(data['conf'][i]) / 100.0  # Convert to 0-1 scale
                
                if text and confidence >= self.confidence_threshold:
                    # Create bounding box in EasyOCR format
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    bbox = [
                        [x, y], [x + w, y],
                        [x + w, y + h], [x, y + h]
                    ]
                    
                    formatted_results.append({
                        'bbox': bbox,
                        'text': text,
                        'confidence': confidence
                    })
            
            logger.debug(f"Tesseract extracted {len(formatted_results)} text regions")
            return formatted_results
            
        except Exception as e:
            logger.debug(f"Tesseract extraction failed (using EasyOCR only): {str(e)}")
            return []
    
    def _extract_card_name(self, ocr_results: List[Dict], image_shape: Tuple) -> Optional[str]:
        """Extract card name from top banner region of card"""
        height, width = image_shape[:2]
        name_candidates = []
        
        for result in ocr_results:
            bbox = result['bbox']
            text = result['text'].strip()
            confidence = result['confidence']
            
            # Calculate center coordinates
            y_center = sum(point[1] for point in bbox) / 4
            x_center = sum(point[0] for point in bbox) / 4
            
            # Card name is in the top banner - expanded to top 25% for off-center cards
            # Lowered confidence requirement for poor quality scans
            if y_center < height * 0.25 and confidence > 0.4:
                # Filter out non-name text patterns
                if (len(text) >= 3 and 
                    not re.match(r'^[\d\{\}/\*\(\)]+$', text) and  # Numbers and symbols only
                    not re.search(r'\billus\b|©|\bcopyright\b|\binc\b|\bllc\b', text, re.IGNORECASE) and  # Artist/copyright indicators
                    not re.search(r'^\d{4,}', text) and  # Long numbers (copyright years, etc)
                    not re.search(r'[™®©]', text) and  # Copyright symbols
                    not text.isnumeric()):  # Pure numbers
                    
                    # Prefer text that's more horizontally centered
                    center_bias = 1.0 - abs(x_center - width/2) / (width/2)
                    score = confidence * center_bias
                    
                    name_candidates.append((text, confidence, y_center, score))
        
        if name_candidates:
            # Sort by position (topmost first), then by score
            name_candidates.sort(key=lambda x: (x[2], -x[3]))
            
            # Return the best candidate from the topmost region
            best_candidate = name_candidates[0]
            logger.debug(f"Extracted card name: '{best_candidate[0]}' (confidence: {best_candidate[1]:.2f})")
            return best_candidate[0]
        
        logger.debug("No valid card name found in top region")
        
        # Fallback: Try to find ANY reasonable text that could be a card name
        return self._fallback_name_extraction(ocr_results, image_shape)
    
    def _fallback_name_extraction(self, ocr_results: List[Dict], image_shape: Tuple) -> Optional[str]:
        """Fallback method to extract any reasonable card name when primary method fails"""
        height, width = image_shape[:2]
        fallback_candidates = []
        
        for result in ocr_results:
            bbox = result['bbox']
            text = result['text'].strip()
            confidence = result['confidence']
            
            # Calculate position
            y_center = sum(point[1] for point in bbox) / 4
            
            # Much more lenient criteria for fallback
            if (confidence > 0.3 and  # Lower confidence threshold
                len(text) >= 2 and   # Shorter minimum length
                y_center < height * 0.4 and  # Top 40% of card
                not re.match(r'^[\d\{\}/\*\(\)]+$', text) and  # Not just symbols
                not text.isnumeric() and  # Not pure number
                not re.search(r'^\d{4,}', text)):  # Not long numbers
                
                # Score based on position and length (prefer topmost, longer text)
                position_score = 1.0 - (y_center / height)
                length_score = min(len(text) / 15, 1.0)  # Prefer longer names up to 15 chars
                total_score = (position_score * 0.6) + (confidence * 0.3) + (length_score * 0.1)
                
                fallback_candidates.append((text, confidence, position_score, total_score))
        
        if fallback_candidates:
            # Sort by total score
            fallback_candidates.sort(key=lambda x: -x[3])
            best_fallback = fallback_candidates[0]
            logger.debug(f"Fallback extracted name: '{best_fallback[0]}' (confidence: {best_fallback[1]:.2f})")
            return best_fallback[0]
        
        logger.debug("No suitable card name found even with fallback")
        return None
    
    def _extract_card_name_targeted(self, card_image: np.ndarray) -> Optional[str]:
        """Extract card name using targeted OCR on title banner region with enhanced accuracy"""
        try:
            height, width = card_image.shape[:2]
            
            # Crop the title banner region with better positioning
            # MTG cards typically have name in top 8-15% of card
            title_region = card_image[int(height * 0.02):int(height * 0.15), int(width * 0.05):int(width * 0.95)]
            
            if title_region.size == 0:
                return None
            
            # Try multiple preprocessing approaches
            processed_regions = []
            processed_regions.append(self._preprocess_title_region(title_region))
            processed_regions.append(self._preprocess_title_region_alternative(title_region))
            
            best_text = None
            best_score = 0
            
            # Try OCR on each processed region
            for processed_title in processed_regions:
                title_ocr_results = self._perform_ocr(processed_title)
                
                if not title_ocr_results:
                    continue
                
                # Find the best text candidate from title region
                for result in title_ocr_results:
                    text = result['text'].strip()
                    confidence = result['confidence']
                    
                    # Enhanced filtering for card names
                    if self._is_valid_card_name_candidate(text, confidence):
                        # Apply additional scoring based on card name characteristics
                        adjusted_score = self._score_card_name_candidate(text, confidence)
                        
                        if adjusted_score > best_score:
                            best_score = adjusted_score
                            best_text = self._clean_card_name(text)
            
            if best_text and best_score > 0.5:
                logger.debug(f"Targeted card name extraction: '{best_text}' (score: {best_score:.2f})")
                return best_text
                
        except Exception as e:
            logger.debug(f"Targeted card name extraction failed: {str(e)}")
        
        return None
    
    def _is_valid_card_name_candidate(self, text: str, confidence: float) -> bool:
        """Enhanced validation for card name candidates"""
        if not text or len(text.strip()) < 2:
            return False
        
        text = text.strip()
        
        # Must have reasonable confidence
        if confidence < 0.4:
            return False
        
        # Filter out obvious non-name patterns
        invalid_patterns = [
            r'^[\d\{\}/\*\(\)\-\.,\s]+$',  # Only symbols/numbers
            r'^[^\w\s]+$',                  # Only special characters
            r'^\s*$',                       # Only whitespace
            r'^\d{1,3}/\d{1,3}$',          # Power/toughness
            r'^\{[WUBRGCTXYS0-9/]+\}$',    # Mana cost only
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, text):
                return False
        
        # Must contain at least one letter
        if not re.search(r'[a-zA-Z]', text):
            return False
        
        # Filter out copyright/artist info
        if re.search(r'\b(illus|©|copyright|tm|wizards|coast)\b', text, re.IGNORECASE):
            return False
        
        # Reasonable length
        if len(text) > 40 or len(text) < 2:
            return False
        
        return True
    
    def _score_card_name_candidate(self, text: str, confidence: float) -> float:
        """Score card name candidates for better selection"""
        score = confidence
        
        # Boost score for text that looks like card names
        text_lower = text.lower()
        
        # Prefer names with title case
        if text.istitle():
            score *= 1.2
        
        # Prefer reasonable length names
        if 3 <= len(text) <= 25:
            score *= 1.1
        
        # Penalty for very short or very long names
        if len(text) < 3 or len(text) > 30:
            score *= 0.8
        
        # Penalty for text with lots of numbers
        num_digits = sum(c.isdigit() for c in text)
        if num_digits > len(text) * 0.3:
            score *= 0.7
        
        # Boost for common MTG name patterns
        if re.search(r'\b(of|the|and|from|to)\b', text_lower):
            score *= 1.1
        
        return score
    
    def _preprocess_title_region_alternative(self, title_image: np.ndarray) -> np.ndarray:
        """Alternative preprocessing approach for title regions"""
        try:
            # Scale up significantly for better character recognition
            height, width = title_image.shape[:2]
            scale_factor = 4 if height < 50 else 2
            scaled = cv2.resize(title_image, (width * scale_factor, height * scale_factor), 
                              interpolation=cv2.INTER_CUBIC)
            
            # Convert to grayscale
            if len(scaled.shape) == 3:
                gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
            else:
                gray = scaled
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Use Otsu's thresholding
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Apply morphological opening to remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            return opened
            
        except Exception as e:
            logger.warning(f"Alternative title preprocessing failed: {str(e)}")
            return title_image
    
    def _clean_card_name(self, name: str) -> str:
        """Clean and normalize extracted card name"""
        if not name:
            return ""
        
        # Remove extra whitespace
        cleaned = ' '.join(name.split())
        
        # Remove leading/trailing punctuation
        cleaned = re.sub(r'^[^\w]+|[^\w]+$', '', cleaned)
        
        # Fix common OCR errors for card names
        ocr_fixes = {
            r'\b0([a-zA-Z])': r'O\1',  # 0 -> O when followed by letter
            r'\b1([a-zA-Z])': r'I\1',  # 1 -> I when followed by letter
            r'([a-zA-Z])0\b': r'\1O',  # 0 -> O when preceded by letter
            r'([a-zA-Z])1\b': r'\1I',  # 1 -> I when preceded by letter
        }
        
        for pattern, replacement in ocr_fixes.items():
            cleaned = re.sub(pattern, replacement, cleaned)
        
        return cleaned.strip()
    
    def _preprocess_title_region(self, title_image: np.ndarray) -> np.ndarray:
        """Specialized preprocessing for card title regions"""
        try:
            # Resize for better OCR - make it larger
            height, width = title_image.shape[:2]
            if height < 80:  # Increased minimum height
                scale_factor = 80 / height
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                title_image = cv2.resize(title_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Convert to grayscale if needed
            if len(title_image.shape) == 3:
                gray = cv2.cvtColor(title_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = title_image
            
            # Apply denoising first
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # Apply adaptive thresholding with different parameters
            binary1 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            binary2 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 8)
            
            # Combine both thresholding results
            combined = cv2.bitwise_or(binary1, binary2)
            
            # Apply morphological operations to clean up text
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
            
            # Additional erosion/dilation to improve text clarity
            kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
            cleaned = cv2.dilate(cleaned, kernel2, iterations=1)
            
            return cleaned
            
        except Exception as e:
            logger.debug(f"Title region preprocessing failed: {str(e)}")
            return title_image
    
    def _extract_mana_cost(self, ocr_results: List[Dict]) -> Optional[str]:
        """Extract and normalize mana cost"""
        all_text = ' '.join([result['text'] for result in ocr_results])
        
        # Find mana cost patterns
        mana_matches = re.findall(self.mana_pattern, all_text)
        
        if mana_matches:
            return ''.join(mana_matches)
        
        # Fallback: look for numbers that might be converted mana cost
        cmc_pattern = r'\b([0-9]{1,2})\b'
        cmc_matches = re.findall(cmc_pattern, all_text)
        
        if cmc_matches:
            # Return the first reasonable CMC found
            for match in cmc_matches:
                cmc = int(match)
                if 0 <= cmc <= 20:  # Reasonable CMC range
                    return f"{{{cmc}}}"
        
        return None
    
    def _extract_type_line(self, ocr_results: List[Dict], image_shape: Tuple) -> Optional[str]:
        """Extract type line from middle region of card"""
        height = image_shape[0]
        type_candidates = []
        
        for result in ocr_results:
            bbox = result['bbox']
            text = result['text']
            confidence = result['confidence']
            
            # Calculate center Y coordinate
            y_center = sum(point[1] for point in bbox) / 4
            
            # Type line is typically in the middle region (25%-60% from top)
            if 0.25 * height < y_center < 0.6 * height:
                # Check if text contains MTG card types
                for card_type, pattern in self.type_patterns.items():
                    if re.search(pattern, text, re.IGNORECASE):
                        type_candidates.append((text, confidence, y_center))
                        break
        
        if type_candidates:
            # Sort by confidence and position
            type_candidates.sort(key=lambda x: (x[1], -x[2]), reverse=True)
            return type_candidates[0][0]
        
        return None
    
    def _extract_oracle_text(self, ocr_results: List[Dict], image_shape: Tuple) -> Optional[str]:
        """Extract rules text from middle-lower region of card"""
        height = image_shape[0]
        text_blocks = []
        
        for result in ocr_results:
            bbox = result['bbox']
            text = result['text']
            confidence = result['confidence']
            
            # Calculate center Y coordinate
            y_center = sum(point[1] for point in bbox) / 4
            
            # Oracle text is typically in the middle-lower region (40%-80% from top)
            if 0.4 * height < y_center < 0.8 * height and confidence > 0.5:
                # Filter out obvious non-rules text
                if len(text) > 3 and not re.match(r'^[\d\{\}/\*]+$', text):
                    text_blocks.append((text, y_center))
        
        if text_blocks:
            # Sort by Y position and combine
            text_blocks.sort(key=lambda x: x[1])
            oracle_text = ' '.join([block[0] for block in text_blocks])
            
            # Clean up the text
            oracle_text = self._clean_oracle_text(oracle_text)
            return oracle_text if len(oracle_text) > 10 else None
        
        return None
    
    def _extract_flavor_text(self, ocr_results: List[Dict], image_shape: Tuple) -> Optional[str]:
        """Extract flavor text (usually italicized) from lower region"""
        height = image_shape[0]
        flavor_candidates = []
        
        for result in ocr_results:
            bbox = result['bbox']
            text = result['text']
            confidence = result['confidence']
            
            # Calculate center Y coordinate
            y_center = sum(point[1] for point in bbox) / 4
            
            # Flavor text is typically in the lower region (70%-90% from top)
            if 0.7 * height < y_center < 0.9 * height and confidence > 0.5:
                # Flavor text often contains quotes or is more descriptive
                if ('"' in text or len(text.split()) > 5) and len(text) > 10:
                    flavor_candidates.append((text, y_center))
        
        if flavor_candidates:
            # Combine all potential flavor text
            flavor_candidates.sort(key=lambda x: x[1])
            flavor_text = ' '.join([candidate[0] for candidate in flavor_candidates])
            return flavor_text.strip()
        
        return None
    
    def _extract_power_toughness(self, ocr_results: List[Dict], image_shape: Tuple) -> Optional[Dict]:
        """Extract power/toughness from bottom-right region"""
        height, width = image_shape[:2]
        pt_candidates = []
        
        for result in ocr_results:
            bbox = result['bbox']
            text = result['text']
            confidence = result['confidence']
            
            # Calculate center coordinates
            x_center = sum(point[0] for point in bbox) / 4
            y_center = sum(point[1] for point in bbox) / 4
            
            # P/T is typically in bottom-right corner
            if (x_center > width * 0.7 and y_center > height * 0.8 and 
                confidence > 0.6):
                
                # Look for P/T pattern
                pt_match = re.search(self.pt_pattern, text)
                if pt_match:
                    power, toughness = pt_match.groups()
                    return {
                        'power': power,
                        'toughness': toughness,
                        'combined': f"{power}/{toughness}"
                    }
                
                # Sometimes OCR splits P/T across multiple detections
                pt_candidates.append((text, x_center, y_center))
        
        # Try to reconstruct P/T from nearby text
        if pt_candidates and len(pt_candidates) >= 2:
            # Sort by position and try to find P/T pattern
            pt_candidates.sort(key=lambda x: (x[2], x[1]))  # Sort by Y then X
            combined_text = ''.join([candidate[0] for candidate in pt_candidates[-3:]])
            
            pt_match = re.search(self.pt_pattern, combined_text)
            if pt_match:
                power, toughness = pt_match.groups()
                return {
                    'power': power,
                    'toughness': toughness,
                    'combined': f"{power}/{toughness}"
                }
        
        return None
    
    def _extract_rarity(self, ocr_results: List[Dict]) -> Optional[str]:
        """Extract rarity information"""
        all_text = ' '.join([result['text'].lower() for result in ocr_results if result.get('text')])
        
        rarity_keywords = {
            'common': ['common', 'c'],
            'uncommon': ['uncommon', 'u'],
            'rare': ['rare', 'r'],
            'mythic': ['mythic', 'mythic rare', 'm']
        }
        
        for rarity, keywords in rarity_keywords.items():
            for keyword in keywords:
                if keyword in all_text:
                    return rarity
        
        return None
    
    def _clean_oracle_text(self, text: str) -> str:
        """Clean and normalize oracle text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR errors in MTG text
        replacements = {
            r'\{T\}': '{T}',  # Tap symbol
            r'\{([WUBRG])\}': r'{\1}',  # Mana symbols
            r'(\d+)/(\d+)': r'\1/\2',  # Power/toughness format
            r'\benter the battlefield\b': 'enters the battlefield',
            r'\bgraveyard\b': 'graveyard',
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _calculate_overall_confidence(self, ocr_results: List[Dict]) -> float:
        """Calculate overall confidence score for the extraction"""
        if not ocr_results:
            return 0.0
        
        confidences = [result['confidence'] for result in ocr_results]
        
        # Weight by text length (longer text blocks are more reliable)
        weighted_confidence = 0
        total_weight = 0
        
        for result in ocr_results:
            weight = len(result['text'])
            weighted_confidence += result['confidence'] * weight
            total_weight += weight
        
        if total_weight > 0:
            return weighted_confidence / total_weight
        else:
            return sum(confidences) / len(confidences)
    
    def _post_process_card_data(self, card_data: Dict) -> Dict:
        """Post-process and validate extracted card data"""
        # Clean card name
        if card_data.get('name'):
            card_data['name'] = self._clean_card_name(card_data['name'])
        
        # Validate and normalize mana cost
        if card_data.get('mana_cost'):
            card_data['converted_mana_cost'] = self._calculate_cmc(card_data['mana_cost'])
        
        # Extract colors from mana cost
        if card_data.get('mana_cost'):
            card_data['colors'] = self._extract_colors(card_data['mana_cost'])
        
        # Validate P/T for creatures
        if card_data.get('type_line') and isinstance(card_data['type_line'], str) and 'creature' in card_data['type_line'].lower():
            if not card_data.get('power_toughness'):
                # Try to extract P/T from oracle text as fallback
                pt_from_text = self._extract_pt_from_text(card_data.get('oracle_text', ''))
                if pt_from_text:
                    card_data['power_toughness'] = pt_from_text
        
        return card_data
    
    def _clean_card_name(self, name: str) -> str:
        """Clean and normalize card name"""
        # Remove common OCR artifacts
        name = re.sub(r'[^\w\s\-\',]', '', name)
        name = re.sub(r'\s+', ' ', name)
        return name.strip()
    
    def _calculate_cmc(self, mana_cost: str) -> int:
        """Calculate converted mana cost"""
        if not mana_cost:
            return 0
        
        cmc = 0
        # Extract numbers from mana cost
        numbers = re.findall(r'\{(\d+)\}', mana_cost)
        for num in numbers:
            cmc += int(num)
        
        # Count colored mana symbols
        colored_symbols = re.findall(r'\{([WUBRG])\}', mana_cost)
        cmc += len(colored_symbols)
        
        # Count hybrid mana (counts as 1 each)
        hybrid_symbols = re.findall(r'\{[WUBRG]/[WUBRG]\}', mana_cost)
        cmc += len(hybrid_symbols)
        
        return cmc
    
    def _extract_colors(self, mana_cost: str) -> str:
        """Extract color identity from mana cost"""
        if not mana_cost:
            return ''
        
        colors = set()
        
        # Find colored mana symbols
        colored_symbols = re.findall(r'\{([WUBRG])\}', mana_cost)
        colors.update(colored_symbols)
        
        # Find hybrid mana symbols
        hybrid_symbols = re.findall(r'\{([WUBRG])/([WUBRG])\}', mana_cost)
        for symbol in hybrid_symbols:
            colors.update(symbol)
        
        # Sort colors in WUBRG order
        color_order = 'WUBRG'
        sorted_colors = ''.join(sorted(colors, key=lambda x: color_order.index(x)))
        
        return sorted_colors
    
    def _extract_pt_from_text(self, text: str) -> Optional[Dict]:
        """Try to extract P/T from oracle text as fallback"""
        if not text:
            return None
        
        # Look for P/T patterns in text
        pt_match = re.search(self.pt_pattern, text)
        if pt_match:
            power, toughness = pt_match.groups()
            return {
                'power': power,
                'toughness': toughness,
                'combined': f"{power}/{toughness}"
            }
        
        return None

class OCRValidator:
    """Validates OCR results against MTG card conventions"""
    
    @staticmethod
    def validate_card_name(name: str) -> bool:
        """Validate that extracted name looks like a real card name"""
        if not name or len(name) < 2:
            return False
        
        # Check for reasonable character composition
        if re.match(r'^[A-Za-z\s\-\',]+$', name):
            return True
        
        return False
    
    @staticmethod
    def validate_mana_cost(mana_cost: str) -> bool:
        """Validate mana cost format"""
        if not mana_cost:
            return True  # Empty mana cost is valid (for lands, etc.)
        
        # Check for valid mana symbols
        valid_pattern = r'^(\{[WUBRGCTXYS0-9]+(?:/[WUBRG])?\})*$'
        return bool(re.match(valid_pattern, mana_cost))
    
    @staticmethod
    def validate_power_toughness(pt_dict: Dict) -> bool:
        """Validate power/toughness values"""
        if not pt_dict:
            return True  # No P/T is valid for non-creatures
        
        power = pt_dict.get('power')
        toughness = pt_dict.get('toughness')
        
        if not power or not toughness:
            return False
        
        # Check if values are valid (numbers or *)
        valid_values = re.match(r'^(\d+|\*)$', power) and re.match(r'^(\d+|\*)$', toughness)
        return valid_values is not None
    
    @staticmethod
    def calculate_confidence_score(card_data: Dict) -> float:
        """Calculate overall confidence score for extracted card data"""
        scores = []
        
        # Name confidence
        if OCRValidator.validate_card_name(card_data.get('name', '')):
            scores.append(0.3)  # 30% weight for valid name
        
        # Mana cost confidence
        if OCRValidator.validate_mana_cost(card_data.get('mana_cost', '')):
            scores.append(0.2)  # 20% weight for valid mana cost
        
        # Type line confidence
        if card_data.get('type_line'):
            scores.append(0.2)  # 20% weight for having type line
        
        # Oracle text confidence
        if card_data.get('oracle_text') and len(card_data['oracle_text']) > 10:
            scores.append(0.2)  # 20% weight for substantial oracle text
        
        # P/T confidence (if creature)
        type_line = (card_data.get('type_line') or '').lower() if isinstance(card_data.get('type_line'), str) else ''
        if 'creature' in type_line:
            if OCRValidator.validate_power_toughness(card_data.get('power_toughness')):
                scores.append(0.1)  # 10% weight for valid P/T
        else:
            scores.append(0.1)  # 10% for non-creatures (no P/T expected)
        
        return sum(scores)
        
        return False
    
    @staticmethod
    def validate_mana_cost(mana_cost: str) -> bool:
        """Validate mana cost format"""
        if not mana_cost:
            return True  # Empty mana cost is valid (for lands, etc.)
        
        # Check for valid mana symbols
        valid_pattern = r'^(\{[WUBRGCTXYS0-9]+(?:/[WUBRG])?\})*$'
        return bool(re.match(valid_pattern, mana_cost))
    
    @staticmethod
    def validate_power_toughness(pt_dict: Dict) -> bool:
        """Validate power/toughness values"""
        if not pt_dict:
            return True  # No P/T is valid for non-creatures
        
        power = pt_dict.get('power')
        toughness = pt_dict.get('toughness')
        
        if not power or not toughness:
            return False
        
        # Check if values are valid (numbers or *)
        valid_values = re.match(r'^(\d+|\*)$', power) and re.match(r'^(\d+|\*)$', toughness)
        return valid_values is not None
        return bool(valid_values)
    
    @staticmethod
    def calculate_confidence_score(card_data: Dict) -> float:
        """Calculate overall confidence score for extracted card data"""
        scores = []
        
        # Name confidence
        if OCRValidator.validate_card_name(card_data.get('name', '')):
            scores.append(0.3)  # 30% weight for valid name
        
        # Mana cost confidence
        if OCRValidator.validate_mana_cost(card_data.get('mana_cost', '')):
            scores.append(0.2)  # 20% weight for valid mana cost
        
        # Type line confidence
        if card_data.get('type_line'):
            scores.append(0.2)  # 20% weight for having type line
        
        # Oracle text confidence
        if card_data.get('oracle_text') and len(card_data['oracle_text']) > 10:
            scores.append(0.2)  # 20% weight for substantial oracle text
        
        # P/T confidence (if creature)
        type_line = (card_data.get('type_line') or '').lower() if isinstance(card_data.get('type_line'), str) else ''
        if 'creature' in type_line:
            if OCRValidator.validate_power_toughness(card_data.get('power_toughness')):
                scores.append(0.1)  # 10% weight for valid P/T
        else:
            scores.append(0.1)  # 10% for non-creatures (no P/T expected)
        
        return sum(scores)
    
    @staticmethod
    def calculate_confidence_score(card_data: Dict) -> float:
        """Calculate overall confidence score for extracted card data"""
        scores = []
        
        # Name confidence
        if OCRValidator.validate_card_name(card_data.get('name', '')):
            scores.append(0.3)  # 30% weight for valid name
        
        # Mana cost confidence
        if OCRValidator.validate_mana_cost(card_data.get('mana_cost', '')):
            scores.append(0.2)  # 20% weight for valid mana cost
        
        # Type line confidence
        if card_data.get('type_line'):
            scores.append(0.2)  # 20% weight for having type line
        
        # Oracle text confidence
        if card_data.get('oracle_text') and len(card_data['oracle_text']) > 10:
            scores.append(0.2)  # 20% weight for substantial oracle text
        
        # P/T confidence (if creature)
        type_line = (card_data.get('type_line') or '').lower() if isinstance(card_data.get('type_line'), str) else ''
        if 'creature' in type_line:
            if OCRValidator.validate_power_toughness(card_data.get('power_toughness')):
                scores.append(0.1)  # 10% weight for valid P/T
        else:
            scores.append(0.1)  # 10% for non-creatures (no P/T expected)
        
        return sum(scores)
        
        return False
    
    @staticmethod
    def validate_mana_cost(mana_cost: str) -> bool:
        """Validate mana cost format"""
        if not mana_cost:
            return True  # Empty mana cost is valid (for lands, etc.)
        
        # Check for valid mana symbols
        valid_pattern = r'^(\{[WUBRGCTXYS0-9]+(?:/[WUBRG])?\})*$'
        return bool(re.match(valid_pattern, mana_cost))
    
    @staticmethod
    def validate_power_toughness(pt_dict: Dict) -> bool:
        """Validate power/toughness values"""
        if not pt_dict:
            return True  # No P/T is valid for non-creatures
        
        power = pt_dict.get('power')
        toughness = pt_dict.get('toughness')
        
        if not power or not toughness:
            return False
        
        # Check if values are valid (numbers or *)
        valid_values = re.match(r'^(\d+|\*)$', power) and re.match(r'^(\d+|\*)$', toughness)
        return valid_values is not None
        return bool(valid_values)
    
    @staticmethod
    def calculate_confidence_score(card_data: Dict) -> float:
        """Calculate overall confidence score for extracted card data"""
        scores = []
        
        # Name confidence
        if OCRValidator.validate_card_name(card_data.get('name', '')):
            scores.append(0.3)  # 30% weight for valid name
        
        # Mana cost confidence
        if OCRValidator.validate_mana_cost(card_data.get('mana_cost', '')):
            scores.append(0.2)  # 20% weight for valid mana cost
        
        # Type line confidence
        if card_data.get('type_line'):
            scores.append(0.2)  # 20% weight for having type line
        
        # Oracle text confidence
        if card_data.get('oracle_text') and len(card_data['oracle_text']) > 10:
            scores.append(0.2)  # 20% weight for substantial oracle text
        
        # P/T confidence (if creature)
        type_line = (card_data.get('type_line') or '').lower() if isinstance(card_data.get('type_line'), str) else ''
        if 'creature' in type_line:
            if OCRValidator.validate_power_toughness(card_data.get('power_toughness')):
                scores.append(0.1)  # 10% weight for valid P/T
        else:
            scores.append(0.1)  # 10% for non-creatures (no P/T expected)
        
        return sum(scores)
        return bool(valid_values)
    
    @staticmethod
    def calculate_confidence_score(card_data: Dict) -> float:
        """Calculate overall confidence score for extracted card data"""
        scores = []
        
        # Name confidence
        if OCRValidator.validate_card_name(card_data.get('name', '')):
            scores.append(0.3)  # 30% weight for valid name
        
        # Mana cost confidence
        if OCRValidator.validate_mana_cost(card_data.get('mana_cost', '')):
            scores.append(0.2)  # 20% weight for valid mana cost
        
        # Type line confidence
        if card_data.get('type_line'):
            scores.append(0.2)  # 20% weight for having type line
        
        # Oracle text confidence
        if card_data.get('oracle_text') and len(card_data['oracle_text']) > 10:
            scores.append(0.2)  # 20% weight for substantial oracle text
        
        # P/T confidence (if creature)
        type_line = (card_data.get('type_line') or '').lower() if isinstance(card_data.get('type_line'), str) else ''
        if 'creature' in type_line:
            if OCRValidator.validate_power_toughness(card_data.get('power_toughness')):
                scores.append(0.1)  # 10% weight for valid P/T
        else:
            scores.append(0.1)  # 10% for non-creatures (no P/T expected)
        
        return sum(scores)
        
        return False
    
    @staticmethod
    def validate_mana_cost(mana_cost: str) -> bool:
        """Validate mana cost format"""
        if not mana_cost:
            return True  # Empty mana cost is valid (for lands, etc.)
        
        # Check for valid mana symbols
        valid_pattern = r'^(\{[WUBRGCTXYS0-9]+(?:/[WUBRG])?\})*$'
        return bool(re.match(valid_pattern, mana_cost))
    
    @staticmethod
    def validate_power_toughness(pt_dict: Dict) -> bool:
        """Validate power/toughness values"""
        if not pt_dict:
            return True  # No P/T is valid for non-creatures
        
        power = pt_dict.get('power')
        toughness = pt_dict.get('toughness')
        
        if not power or not toughness:
            return False
        
        # Check if values are valid (numbers or *)
        valid_values = re.match(r'^(\d+|\*)$', power) and re.match(r'^(\d+|\*)$', toughness)
        return valid_values is not None
        return bool(valid_values)
    
    @staticmethod
    def calculate_confidence_score(card_data: Dict) -> float:
        """Calculate overall confidence score for extracted card data"""
        scores = []
        
        # Name confidence
        if OCRValidator.validate_card_name(card_data.get('name', '')):
            scores.append(0.3)  # 30% weight for valid name
        
        # Mana cost confidence
        if OCRValidator.validate_mana_cost(card_data.get('mana_cost', '')):
            scores.append(0.2)  # 20% weight for valid mana cost
        
        # Type line confidence
        if card_data.get('type_line'):
            scores.append(0.2)  # 20% weight for having type line
        
        # Oracle text confidence
        if card_data.get('oracle_text') and len(card_data['oracle_text']) > 10:
            scores.append(0.2)  # 20% weight for substantial oracle text
        
        # P/T confidence (if creature)
        type_line = (card_data.get('type_line') or '').lower() if isinstance(card_data.get('type_line'), str) else ''
        if 'creature' in type_line:
            if OCRValidator.validate_power_toughness(card_data.get('power_toughness')):
                scores.append(0.1)  # 10% weight for valid P/T
        else:
            scores.append(0.1)  # 10% for non-creatures (no P/T expected)
        
        return sum(scores)
        
        if text_blocks:
            # Sort by Y position and combine
            text_blocks.sort(key=lambda x: x[1])
            oracle_text = ' '.join([block[0] for block in text_blocks])
            
            # Clean up the text
            oracle_text = self._clean_oracle_text(oracle_text)
            return oracle_text if len(oracle_text) > 10 else None
        
        return None
    
    def _extract_flavor_text(self, ocr_results: List[Dict], image_shape: Tuple) -> Optional[str]:
        """Extract flavor text (usually italicized) from lower region"""
        height = image_shape[0]
        flavor_candidates = []
        
        for result in ocr_results:
            bbox = result['bbox']
            text = result['text']
            confidence = result['confidence']
            
            # Calculate center Y coordinate
            y_center = sum(point[1] for point in bbox) / 4
            
            # Flavor text is typically in the lower region (70%-90% from top)
            if 0.7 * height < y_center < 0.9 * height and confidence > 0.5:
                # Flavor text often contains quotes or is more descriptive
                if ('"' in text or len(text.split()) > 5) and len(text) > 10:
                    flavor_candidates.append((text, y_center))
        
        if flavor_candidates:
            # Combine all potential flavor text
            flavor_candidates.sort(key=lambda x: x[1])
            flavor_text = ' '.join([candidate[0] for candidate in flavor_candidates])
            return flavor_text.strip()
        
        return None
    
    def _extract_power_toughness(self, ocr_results: List[Dict], image_shape: Tuple) -> Optional[Dict]:
        """Extract power/toughness from bottom-right region"""
        height, width = image_shape[:2]
        pt_candidates = []
        
        for result in ocr_results:
            bbox = result['bbox']
            text = result['text']
            confidence = result['confidence']
            
            # Calculate center coordinates
            x_center = sum(point[0] for point in bbox) / 4
            y_center = sum(point[1] for point in bbox) / 4
            
            # P/T is typically in bottom-right corner
            if (x_center > width * 0.7 and y_center > height * 0.8 and 
                confidence > 0.6):
                
                # Look for P/T pattern
                pt_match = re.search(self.pt_pattern, text)
                if pt_match:
                    power, toughness = pt_match.groups()
                    return {
                        'power': power,
                        'toughness': toughness,
                        'combined': f"{power}/{toughness}"
                    }
                
                # Sometimes OCR splits P/T across multiple detections
                pt_candidates.append((text, x_center, y_center))
        
        # Try to reconstruct P/T from nearby text
        if pt_candidates and len(pt_candidates) >= 2:
            # Sort by position and try to find P/T pattern
            pt_candidates.sort(key=lambda x: (x[2], x[1]))  # Sort by Y then X
            combined_text = ''.join([candidate[0] for candidate in pt_candidates[-3:]])
            
            pt_match = re.search(self.pt_pattern, combined_text)
            if pt_match:
                power, toughness = pt_match.groups()
                return {
                    'power': power,
                    'toughness': toughness,
                    'combined': f"{power}/{toughness}"
                }
        
        return None
    
    def _extract_rarity(self, ocr_results: List[Dict]) -> Optional[str]:
        """Extract rarity information"""
        all_text = ' '.join([result['text'].lower() for result in ocr_results if result.get('text')])
        
        rarity_keywords = {
            'common': ['common', 'c'],
            'uncommon': ['uncommon', 'u'],
            'rare': ['rare', 'r'],
            'mythic': ['mythic', 'mythic rare', 'm']
        }
        
        for rarity, keywords in rarity_keywords.items():
            for keyword in keywords:
                if keyword in all_text:
                    return rarity
        
        return None
    
    def _clean_oracle_text(self, text: str) -> str:
        """Clean and normalize oracle text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR errors in MTG text
        replacements = {
            r'\{T\}': '{T}',  # Tap symbol
            r'\{([WUBRG])\}': r'{\1}',  # Mana symbols
            r'(\d+)/(\d+)': r'\1/\2',  # Power/toughness format
            r'\benter the battlefield\b': 'enters the battlefield',
            r'\bgraveyard\b': 'graveyard',
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _calculate_overall_confidence(self, ocr_results: List[Dict]) -> float:
        """Calculate overall confidence score for the extraction"""
        if not ocr_results:
            return 0.0
        
        confidences = [result['confidence'] for result in ocr_results]
        
        # Weight by text length (longer text blocks are more reliable)
        weighted_confidence = 0
        total_weight = 0
        
        for result in ocr_results:
            weight = len(result['text'])
            weighted_confidence += result['confidence'] * weight
            total_weight += weight
        
        if total_weight > 0:
            return weighted_confidence / total_weight
        else:
            return sum(confidences) / len(confidences)
    
    def _post_process_card_data(self, card_data: Dict) -> Dict:
        """Post-process and validate extracted card data"""
        # Clean card name
        if card_data.get('name'):
            card_data['name'] = self._clean_card_name(card_data['name'])
        
        # Validate and normalize mana cost
        if card_data.get('mana_cost'):
            card_data['converted_mana_cost'] = self._calculate_cmc(card_data['mana_cost'])
        
        # Extract colors from mana cost
        if card_data.get('mana_cost'):
            card_data['colors'] = self._extract_colors(card_data['mana_cost'])
        
        # Validate P/T for creatures
        if card_data.get('type_line') and isinstance(card_data['type_line'], str) and 'creature' in card_data['type_line'].lower():
            if not card_data.get('power_toughness'):
                # Try to extract P/T from oracle text as fallback
                pt_from_text = self._extract_pt_from_text(card_data.get('oracle_text', ''))
                if pt_from_text:
                    card_data['power_toughness'] = pt_from_text
        
        return card_data
    
    def _clean_card_name(self, name: str) -> str:
        """Clean and normalize card name"""
        # Remove common OCR artifacts
        name = re.sub(r'[^\w\s\-\',]', '', name)
        name = re.sub(r'\s+', ' ', name)
        return name.strip()
    
    def _calculate_cmc(self, mana_cost: str) -> int:
        """Calculate converted mana cost"""
        if not mana_cost:
            return 0
        
        cmc = 0
        # Extract numbers from mana cost
        numbers = re.findall(r'\{(\d+)\}', mana_cost)
        for num in numbers:
            cmc += int(num)
        
        # Count colored mana symbols
        colored_symbols = re.findall(r'\{([WUBRG])\}', mana_cost)
        cmc += len(colored_symbols)
        
        # Count hybrid mana (counts as 1 each)
        hybrid_symbols = re.findall(r'\{[WUBRG]/[WUBRG]\}', mana_cost)
        cmc += len(hybrid_symbols)
        
        return cmc
    
    def _extract_colors(self, mana_cost: str) -> str:
        """Extract color identity from mana cost"""
        if not mana_cost:
            return ''
        
        colors = set()
        
        # Find colored mana symbols
        colored_symbols = re.findall(r'\{([WUBRG])\}', mana_cost)
        colors.update(colored_symbols)
        
        # Find hybrid mana symbols
        hybrid_symbols = re.findall(r'\{([WUBRG])/([WUBRG])\}', mana_cost)
        for symbol in hybrid_symbols:
            colors.update(symbol)
        
        # Sort colors in WUBRG order
        color_order = 'WUBRG'
        sorted_colors = ''.join(sorted(colors, key=lambda x: color_order.index(x)))
        
        return sorted_colors
    
    def _extract_pt_from_text(self, text: str) -> Optional[Dict]:
        """Try to extract P/T from oracle text as fallback"""
        if not text:
            return None
        
        # Look for P/T patterns in text
        pt_match = re.search(self.pt_pattern, text)
        if pt_match:
            power, toughness = pt_match.groups()
            return {
                'power': power,
                'toughness': toughness,
                'combined': f"{power}/{toughness}"
            }
        
        return None

class OCRValidator:
    """Validates OCR results against MTG card conventions"""
    
    @staticmethod
    def validate_card_name(name: str) -> bool:
        """Validate that extracted name looks like a real card name"""
        if not name or len(name) < 2:
            return False
        
        # Check for reasonable character composition
        if re.match(r'^[A-Za-z\s\-\',]+$', name):
            return True
        
        return False
    
    @staticmethod
    def validate_mana_cost(mana_cost: str) -> bool:
        """Validate mana cost format"""
        if not mana_cost:
            return True  # Empty mana cost is valid (for lands, etc.)
        
        # Check for valid mana symbols
        valid_pattern = r'^(\{[WUBRGCTXYS0-9]+(?:/[WUBRG])?\})*$'
        return bool(re.match(valid_pattern, mana_cost))
    
    @staticmethod
    def validate_power_toughness(pt_dict: Dict) -> bool:
        """Validate power/toughness values"""
        if not pt_dict:
            return True  # No P/T is valid for non-creatures
        
        power = pt_dict.get('power')
        toughness = pt_dict.get('toughness')
        
        if not power or not toughness:
            return False
        
        # Check if values are valid (numbers or *)
        valid_values = re.match(r'^(\d+|\*)$', power) and re.match(r'^(\d+|\*)$', toughness)
        return valid_values is not None
    
    @staticmethod
    def calculate_confidence_score(card_data: Dict) -> float:
        """Calculate overall confidence score for extracted card data"""
        scores = []
        
        # Name confidence
        if OCRValidator.validate_card_name(card_data.get('name', '')):
            scores.append(0.3)  # 30% weight for valid name
        
        # Mana cost confidence
        if OCRValidator.validate_mana_cost(card_data.get('mana_cost', '')):
            scores.append(0.2)  # 20% weight for valid mana cost
        
        # Type line confidence
        if card_data.get('type_line'):
            scores.append(0.2)  # 20% weight for having type line
        
        # Oracle text confidence
        if card_data.get('oracle_text') and len(card_data['oracle_text']) > 10:
            scores.append(0.2)  # 20% weight for substantial oracle text
        
        # P/T confidence (if creature)
        type_line = (card_data.get('type_line') or '').lower() if isinstance(card_data.get('type_line'), str) else ''
        if 'creature' in type_line:
            if OCRValidator.validate_power_toughness(card_data.get('power_toughness')):
                scores.append(0.1)  # 10% weight for valid P/T
        else:
            scores.append(0.1)  # 10% for non-creatures (no P/T expected)
        
        return sum(scores)
        
        return False
    
    @staticmethod
    def validate_mana_cost(mana_cost: str) -> bool:
        """Validate mana cost format"""
        if not mana_cost:
            return True  # Empty mana cost is valid (for lands, etc.)
        
        # Check for valid mana symbols
        valid_pattern = r'^(\{[WUBRGCTXYS0-9]+(?:/[WUBRG])?\})*$'
        return bool(re.match(valid_pattern, mana_cost))
    
    @staticmethod
    def validate_power_toughness(pt_dict: Dict) -> bool:
        """Validate power/toughness values"""
        if not pt_dict:
            return True  # No P/T is valid for non-creatures
        
        power = pt_dict.get('power')
        toughness = pt_dict.get('toughness')
        
        if not power or not toughness:
            return False
        
        # Check if values are valid (numbers or *)
        valid_values = re.match(r'^(\d+|\*)$', power) and re.match(r'^(\d+|\*)$', toughness)
        return valid_values is not None
        return bool(valid_values)
    
    @staticmethod
    def calculate_confidence_score(card_data: Dict) -> float:
        """Calculate overall confidence score for extracted card data"""
        scores = []
        
        # Name confidence
        if OCRValidator.validate_card_name(card_data.get('name', '')):
            scores.append(0.3)  # 30% weight for valid name
        
        # Mana cost confidence
        if OCRValidator.validate_mana_cost(card_data.get('mana_cost', '')):
            scores.append(0.2)  # 20% weight for valid mana cost
        
        # Type line confidence
        if card_data.get('type_line'):
            scores.append(0.2)  # 20% weight for having type line
        
        # Oracle text confidence
        if card_data.get('oracle_text') and len(card_data['oracle_text']) > 10:
            scores.append(0.2)  # 20% weight for substantial oracle text
        
        # P/T confidence (if creature)
        type_line = (card_data.get('type_line') or '').lower() if isinstance(card_data.get('type_line'), str) else ''
        if 'creature' in type_line:
            if OCRValidator.validate_power_toughness(card_data.get('power_toughness')):
                scores.append(0.1)  # 10% weight for valid P/T
        else:
            scores.append(0.1)  # 10% for non-creatures (no P/T expected)
        
        return sum(scores)
    
    @staticmethod
    def calculate_confidence_score(card_data: Dict) -> float:
        """Calculate overall confidence score for extracted card data"""
        scores = []
        
        # Name confidence
        if OCRValidator.validate_card_name(card_data.get('name', '')):
            scores.append(0.3)  # 30% weight for valid name
        
        # Mana cost confidence
        if OCRValidator.validate_mana_cost(card_data.get('mana_cost', '')):
            scores.append(0.2)  # 20% weight for valid mana cost
        
        # Type line confidence
        if card_data.get('type_line'):
            scores.append(0.2)  # 20% weight for having type line
        
        # Oracle text confidence
        if card_data.get('oracle_text') and len(card_data['oracle_text']) > 10:
            scores.append(0.2)  # 20% weight for substantial oracle text
        
        # P/T confidence (if creature)
        type_line = (card_data.get('type_line') or '').lower() if isinstance(card_data.get('type_line'), str) else ''
        if 'creature' in type_line:
            if OCRValidator.validate_power_toughness(card_data.get('power_toughness')):
                scores.append(0.1)  # 10% weight for valid P/T
        else:
            scores.append(0.1)  # 10% for non-creatures (no P/T expected)
        
        return sum(scores)
        
        return False
    
    @staticmethod
    def validate_mana_cost(mana_cost: str) -> bool:
        """Validate mana cost format"""
        if not mana_cost:
            return True  # Empty mana cost is valid (for lands, etc.)
        
        # Check for valid mana symbols
        valid_pattern = r'^(\{[WUBRGCTXYS0-9]+(?:/[WUBRG])?\})*$'
        return bool(re.match(valid_pattern, mana_cost))
    
    @staticmethod
    def validate_power_toughness(pt_dict: Dict) -> bool:
        """Validate power/toughness values"""
        if not pt_dict:
            return True  # No P/T is valid for non-creatures
        
        power = pt_dict.get('power')
        toughness = pt_dict.get('toughness')
        
        if not power or not toughness:
            return False
        
        # Check if values are valid (numbers or *)
        valid_values = re.match(r'^(\d+|\*)$', power) and re.match(r'^(\d+|\*)$', toughness)
        return valid_values is not None
        return bool(valid_values)
    
    @staticmethod
    def calculate_confidence_score(card_data: Dict) -> float:
        """Calculate overall confidence score for extracted card data"""
        scores = []
        
        # Name confidence
        if OCRValidator.validate_card_name(card_data.get('name', '')):
            scores.append(0.3)  # 30% weight for valid name
        
        # Mana cost confidence
        if OCRValidator.validate_mana_cost(card_data.get('mana_cost', '')):
            scores.append(0.2)  # 20% weight for valid mana cost
        
        # Type line confidence
        if card_data.get('type_line'):
            scores.append(0.2)  # 20% weight for having type line
        
        # Oracle text confidence
        if card_data.get('oracle_text') and len(card_data['oracle_text']) > 10:
            scores.append(0.2)  # 20% weight for substantial oracle text
        
        # P/T confidence (if creature)
        type_line = (card_data.get('type_line') or '').lower() if isinstance(card_data.get('type_line'), str) else ''
        if 'creature' in type_line:
            if OCRValidator.validate_power_toughness(card_data.get('power_toughness')):
                scores.append(0.1)  # 10% weight for valid P/T
        else:
            scores.append(0.1)  # 10% for non-creatures (no P/T expected)
        
        return sum(scores)
        return bool(valid_values)
    
    @staticmethod
    def calculate_confidence_score(card_data: Dict) -> float:
        """Calculate overall confidence score for extracted card data"""
        scores = []
        
        # Name confidence
        if OCRValidator.validate_card_name(card_data.get('name', '')):
            scores.append(0.3)  # 30% weight for valid name
        
        # Mana cost confidence
        if OCRValidator.validate_mana_cost(card_data.get('mana_cost', '')):
            scores.append(0.2)  # 20% weight for valid mana cost
        
        # Type line confidence
        if card_data.get('type_line'):
            scores.append(0.2)  # 20% weight for having type line
        
        # Oracle text confidence
        if card_data.get('oracle_text') and len(card_data['oracle_text']) > 10:
            scores.append(0.2)  # 20% weight for substantial oracle text
        
        # P/T confidence (if creature)
        type_line = (card_data.get('type_line') or '').lower() if isinstance(card_data.get('type_line'), str) else ''
        if 'creature' in type_line:
            if OCRValidator.validate_power_toughness(card_data.get('power_toughness')):
                scores.append(0.1)  # 10% weight for valid P/T
        else:
            scores.append(0.1)  # 10% for non-creatures (no P/T expected)
        
        return sum(scores)
        
        return False
    
    @staticmethod
    def validate_mana_cost(mana_cost: str) -> bool:
        """Validate mana cost format"""
        if not mana_cost:
            return True  # Empty mana cost is valid (for lands, etc.)
        
        # Check for valid mana symbols
        valid_pattern = r'^(\{[WUBRGCTXYS0-9]+(?:/[WUBRG])?\})*$'
        return bool(re.match(valid_pattern, mana_cost))
    
    @staticmethod
    def validate_power_toughness(pt_dict: Dict) -> bool:
        """Validate power/toughness values"""
        if not pt_dict:
            return True  # No P/T is valid for non-creatures
        
        power = pt_dict.get('power')
        toughness = pt_dict.get('toughness')
        
        if not power or not toughness:
            return False
        
        # Check if values are valid (numbers or *)
        valid_values = re.match(r'^(\d+|\*)$', power) and re.match(r'^(\d+|\*)$', toughness)
        return valid_values is not None
        return bool(valid_values)
    
    @staticmethod
    def calculate_confidence_score(card_data: Dict) -> float:
        """Calculate overall confidence score for extracted card data"""
        scores = []
        
        # Name confidence
        if OCRValidator.validate_card_name(card_data.get('name', '')):
            scores.append(0.3)  # 30% weight for valid name
        
        # Mana cost confidence
        if OCRValidator.validate_mana_cost(card_data.get('mana_cost', '')):
            scores.append(0.2)  # 20% weight for valid mana cost
        
        # Type line confidence
        if card_data.get('type_line'):
            scores.append(0.2)  # 20% weight for having type line
        
        # Oracle text confidence
        if card_data.get('oracle_text') and len(card_data['oracle_text']) > 10:
            scores.append(0.2)  # 20% weight for substantial oracle text
        
        # P/T confidence (if creature)
        type_line = (card_data.get('type_line') or '').lower() if isinstance(card_data.get('type_line'), str) else ''
        if 'creature' in type_line:
            if OCRValidator.validate_power_toughness(card_data.get('power_toughness')):
                scores.append(0.1)  # 10% weight for valid P/T
        else:
            scores.append(0.1)  # 10% for non-creatures (no P/T expected)
        
        return sum(scores)  # Not just mana symbols/numbers
    
    def _extract_flavor_text(self, ocr_results: List[Dict], image_shape: Tuple) -> Optional[str]:
        """Extract flavor text (usually italicized) from lower region"""
        height = image_shape[0]
        top_region_texts = []
        
        for result in ocr_results:
            bbox = result['bbox']
            text = result['text'].strip()
            confidence = result['confidence']
            
            # Calculate center position
            y_center = sum(point[1] for point in bbox) / 4
            
            # Flavor text is typically in the lower region (70%-90% from top)
            if 0.7 * height < y_center < 0.9 * height and confidence > 0.4:
                # Look for italic-like patterns or obvious flavor text
                if len(text) > 5:
                    top_region_texts.append((text, confidence, y_center))
        
        if top_region_texts:
            # Sort by Y position (higher on card = lower Y value)
            top_region_texts.sort(key=lambda x: x[2])
            
            # Take the highest confidence text from the topmost region
            best_candidate = max(top_region_texts[:3], key=lambda x: x[1])
            return best_candidate[0]
        
        return None
    
    def _extract_mana_cost(self, ocr_results: List[Dict]) -> Optional[str]:
        """Extract and normalize mana cost"""
        all_text = ' '.join([result['text'] for result in ocr_results])
        
        # Find mana cost patterns
        mana_matches = re.findall(self.mana_pattern, all_text)
        
        if mana_matches:
            return ''.join(mana_matches)
        
        # Fallback: look for numbers that might be converted mana cost
        cmc_pattern = r'\b([0-9]{1,2})\b'
        cmc_matches = re.findall(cmc_pattern, all_text)
        
        if cmc_matches:
            # Return the first reasonable CMC found
            for match in cmc_matches:
                cmc = int(match)
                if 0 <= cmc <= 20:  # Reasonable CMC range
                    return f"{{{cmc}}}"
        
        return None
    
    def _extract_type_line(self, ocr_results: List[Dict], image_shape: Tuple) -> Optional[str]:
        """Extract type line from middle region of card"""
        height = image_shape[0]
        type_candidates = []
        
        for result in ocr_results:
            bbox = result['bbox']
            text = result['text']
            confidence = result['confidence']
            
            # Calculate center Y coordinate
            y_center = sum(point[1] for point in bbox) / 4
            
            # Type line is typically in the middle region (25%-60% from top)
            if 0.25 * height < y_center < 0.6 * height:
                # Check if text contains MTG card types
                for card_type, pattern in self.type_patterns.items():
                    if re.search(pattern, text, re.IGNORECASE):
                        type_candidates.append((text, confidence, y_center))
                        break
        
        if type_candidates:
            # Sort by confidence and position
            type_candidates.sort(key=lambda x: (x[1], -x[2]), reverse=True)
            return type_candidates[0][0]
        
        return None
    
    def _extract_oracle_text(self, ocr_results: List[Dict], image_shape: Tuple) -> Optional[str]:
        """Extract rules text from middle-lower region of card"""
        height = image_shape[0]
        text_blocks = []
        
        for result in ocr_results:
            bbox = result['bbox']
            text = result['text']
            confidence = result['confidence']
            
            # Calculate center Y coordinate
            y_center = sum(point[1] for point in bbox) / 4
            
            # Oracle text is typically in the middle-lower region (40%-80% from top)
            if 0.4 * height < y_center < 0.8 * height and confidence > 0.5:
                # Filter out obvious non-rules text
                if len(text) > 3 and not re.match(r'^[\d\{\}/\*]+$', text):
                    text_blocks.append((text, y_center))
        
        if text_blocks:
            # Sort by Y position and combine
            text_blocks.sort(key=lambda x: x[1])
            oracle_text = ' '.join([block[0] for block in text_blocks])
            
            # Clean up the text
            oracle_text = self._clean_oracle_text(oracle_text)
            return oracle_text if len(oracle_text) > 10 else None
        
        return None
    
    def _extract_flavor_text(self, ocr_results: List[Dict], image_shape: Tuple) -> Optional[str]:
        """Extract flavor text (usually italicized) from lower region"""
        height = image_shape[0]
        flavor_candidates = []
        
        for result in ocr_results:
            bbox = result['bbox']
            text = result['text']
            confidence = result['confidence']
            
            # Calculate center Y coordinate
            y_center = sum(point[1] for point in bbox) / 4
            
            # Flavor text is typically in the lower region (70%-90% from top)
            if 0.7 * height < y_center < 0.9 * height and confidence > 0.5:
                # Flavor text often contains quotes or is more descriptive
                if ('"' in text or len(text.split()) > 5) and len(text) > 10:
                    flavor_candidates.append((text, y_center))
        
        if flavor_candidates:
            # Combine all potential flavor text
            flavor_candidates.sort(key=lambda x: x[1])
            flavor_text = ' '.join([candidate[0] for candidate in flavor_candidates])
            return flavor_text.strip()
        
        return None
    
    def _extract_power_toughness(self, ocr_results: List[Dict], image_shape: Tuple) -> Optional[Dict]:
        """Extract power/toughness from bottom-right region"""
        height, width = image_shape[:2]
        pt_candidates = []
        
        for result in ocr_results:
            bbox = result['bbox']
            text = result['text']
            confidence = result['confidence']
            
            # Calculate center coordinates
            x_center = sum(point[0] for point in bbox) / 4
            y_center = sum(point[1] for point in bbox) / 4
            
            # P/T is typically in bottom-right corner
            if (x_center > width * 0.7 and y_center > height * 0.8 and 
                confidence > 0.6):
                
                # Look for P/T pattern
                pt_match = re.search(self.pt_pattern, text)
                if pt_match:
                    power, toughness = pt_match.groups()
                    return {
                        'power': power,
                        'toughness': toughness,
                        'combined': f"{power}/{toughness}"
                    }
                
                # Sometimes OCR splits P/T across multiple detections
                pt_candidates.append((text, x_center, y_center))
        
        # Try to reconstruct P/T from nearby text
        if pt_candidates and len(pt_candidates) >= 2:
            # Sort by position and try to find P/T pattern
            pt_candidates.sort(key=lambda x: (x[2], x[1]))  # Sort by Y then X
            combined_text = ''.join([candidate[0] for candidate in pt_candidates[-3:]])
            
            pt_match = re.search(self.pt_pattern, combined_text)
            if pt_match:
                power, toughness = pt_match.groups()
                return {
                    'power': power,
                    'toughness': toughness,
                    'combined': f"{power}/{toughness}"
                }
        
        return None
    
    def _extract_rarity(self, ocr_results: List[Dict]) -> Optional[str]:
        """Extract rarity information"""
        all_text = ' '.join([result['text'].lower() for result in ocr_results if result.get('text')])
        
        rarity_keywords = {
            'common': ['common', 'c'],
            'uncommon': ['uncommon', 'u'],
            'rare': ['rare', 'r'],
            'mythic': ['mythic', 'mythic rare', 'm']
        }
        
        for rarity, keywords in rarity_keywords.items():
            for keyword in keywords:
                if keyword in all_text:
                    return rarity
        
        return None
    
    def _clean_oracle_text(self, text: str) -> str:
        """Clean and normalize oracle text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR errors in MTG text
        replacements = {
            r'\{T\}': '{T}',  # Tap symbol
            r'\{([WUBRG])\}': r'{\1}',  # Mana symbols
            r'(\d+)/(\d+)': r'\1/\2',  # Power/toughness format
            r'\benter the battlefield\b': 'enters the battlefield',
            r'\bgraveyard\b': 'graveyard',
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _calculate_overall_confidence(self, ocr_results: List[Dict]) -> float:
        """Calculate overall confidence score for the extraction"""
        if not ocr_results:
            return 0.0
        
        confidences = [result['confidence'] for result in ocr_results]
        
        # Weight by text length (longer text blocks are more reliable)
        weighted_confidence = 0
        total_weight = 0
        
        for result in ocr_results:
            weight = len(result['text'])
            weighted_confidence += result['confidence'] * weight
            total_weight += weight
        
        if total_weight > 0:
            return weighted_confidence / total_weight
        else:
            return sum(confidences) / len(confidences)
    
    def _post_process_card_data(self, card_data: Dict) -> Dict:
        """Post-process and validate extracted card data"""
        # Clean card name
        if card_data.get('name'):
            card_data['name'] = self._clean_card_name(card_data['name'])
        
        # Validate and normalize mana cost
        if card_data.get('mana_cost'):
            card_data['converted_mana_cost'] = self._calculate_cmc(card_data['mana_cost'])
        
        # Extract colors from mana cost
        if card_data.get('mana_cost'):
            card_data['colors'] = self._extract_colors(card_data['mana_cost'])
        
        # Validate P/T for creatures
        if card_data.get('type_line') and isinstance(card_data['type_line'], str) and 'creature' in card_data['type_line'].lower():
            if not card_data.get('power_toughness'):
                # Try to extract P/T from oracle text as fallback
                pt_from_text = self._extract_pt_from_text(card_data.get('oracle_text', ''))
                if pt_from_text:
                    card_data['power_toughness'] = pt_from_text
        
        return card_data
    
    def _clean_card_name(self, name: str) -> str:
        """Clean and normalize card name"""
        # Remove common OCR artifacts
        name = re.sub(r'[^\w\s\-\',]', '', name)
        name = re.sub(r'\s+', ' ', name)
        return name.strip()
    
    def _calculate_cmc(self, mana_cost: str) -> int:
        """Calculate converted mana cost"""
        if not mana_cost:
            return 0
        
        cmc = 0
        # Extract numbers from mana cost
        numbers = re.findall(r'\{(\d+)\}', mana_cost)
        for num in numbers:
            cmc += int(num)
        
        # Count colored mana symbols
        colored_symbols = re.findall(r'\{([WUBRG])\}', mana_cost)
        cmc += len(colored_symbols)
        
        # Count hybrid mana (counts as 1 each)
        hybrid_symbols = re.findall(r'\{[WUBRG]/[WUBRG]\}', mana_cost)
        cmc += len(hybrid_symbols)
        
        return cmc
    
    def _extract_colors(self, mana_cost: str) -> str:
        """Extract color identity from mana cost"""
        if not mana_cost:
            return ''
        
        colors = set()
        
        # Find colored mana symbols
        colored_symbols = re.findall(r'\{([WUBRG])\}', mana_cost)
        colors.update(colored_symbols)
        
        # Find hybrid mana symbols
        hybrid_symbols = re.findall(r'\{([WUBRG])/([WUBRG])\}', mana_cost)
        for symbol in hybrid_symbols:
            colors.update(symbol)
        
        # Sort colors in WUBRG order
        color_order = 'WUBRG'
        sorted_colors = ''.join(sorted(colors, key=lambda x: color_order.index(x)))
        
        return sorted_colors
    
    def _extract_pt_from_text(self, text: str) -> Optional[Dict]:
        """Try to extract P/T from oracle text as fallback"""
        if not text:
            return None
        
        # Look for P/T patterns in text
        pt_match = re.search(self.pt_pattern, text)
        if pt_match:
            power, toughness = pt_match.groups()
            return {
                'power': power,
                'toughness': toughness,
                'combined': f"{power}/{toughness}"
            }
        
        return None

class OCRValidator:
    """Validates OCR results against MTG card conventions"""
    
    @staticmethod
    def validate_card_name(name: str) -> bool:
        """Validate that extracted name looks like a real card name"""
        if not name or len(name) < 2:
            return False
        
        # Check for reasonable character composition
        if re.match(r'^[A-Za-z\s\-\',]+$', name):
            return True
        
        return False
    
    @staticmethod
    def validate_mana_cost(mana_cost: str) -> bool:
        """Validate mana cost format"""
        if not mana_cost:
            return True  # Empty mana cost is valid (for lands, etc.)
        
        # Check for valid mana symbols
        valid_pattern = r'^(\{[WUBRGCTXYS0-9]+(?:/[WUBRG])?\})*$'
        return bool(re.match(valid_pattern, mana_cost))
    
    @staticmethod
    def validate_power_toughness(pt_dict: Dict) -> bool:
        """Validate power/toughness values"""
        if not pt_dict:
            return True  # No P/T is valid for non-creatures
        
        power = pt_dict.get('power')
        toughness = pt_dict.get('toughness')
        
        if not power or not toughness:
            return False
        
        # Check if values are valid (numbers or *)
        valid_values = re.match(r'^(\d+|\*)$', power) and re.match(r'^(\d+|\*)$', toughness)
        return valid_values is not None
    
    @staticmethod
    def calculate_confidence_score(card_data: Dict) -> float:
        """Calculate overall confidence score for extracted card data"""
        scores = []
        
        # Name confidence
        if OCRValidator.validate_card_name(card_data.get('name', '')):
            scores.append(0.3)  # 30% weight for valid name
        
        # Mana cost confidence
        if OCRValidator.validate_mana_cost(card_data.get('mana_cost', '')):
            scores.append(0.2)  # 20% weight for valid mana cost
        
        # Type line confidence
        if card_data.get('type_line'):
            scores.append(0.2)  # 20% weight for having type line
        
        # Oracle text confidence
        if card_data.get('oracle_text') and len(card_data['oracle_text']) > 10:
            scores.append(0.2)  # 20% weight for substantial oracle text
        
        # P/T confidence (if creature)
        type_line = (card_data.get('type_line') or '').lower() if isinstance(card_data.get('type_line'), str) else ''
        if 'creature' in type_line:
            if OCRValidator.validate_power_toughness(card_data.get('power_toughness')):
                scores.append(0.1)  # 10% weight for valid P/T
        else:
            scores.append(0.1)  # 10% for non-creatures (no P/T expected)
        
        return sum(scores)
        
        return False
    
    @staticmethod
    def validate_mana_cost(mana_cost: str) -> bool:
        """Validate mana cost format"""
        if not mana_cost:
            return True  # Empty mana cost is valid (for lands, etc.)
        
        # Check for valid mana symbols
        valid_pattern = r'^(\{[WUBRGCTXYS0-9]+(?:/[WUBRG])?\})*$'
        return bool(re.match(valid_pattern, mana_cost))
    
    @staticmethod
    def validate_power_toughness(pt_dict: Dict) -> bool:
        """Validate power/toughness values"""
        if not pt_dict:
            return True  # No P/T is valid for non-creatures
        
        power = pt_dict.get('power')
        toughness = pt_dict.get('toughness')
        
        if not power or not toughness:
            return False
        
        # Check if values are valid (numbers or *)
        valid_values = re.match(r'^(\d+|\*)$', power) and re.match(r'^(\d+|\*)$', toughness)
        return valid_values is not None
        return bool(valid_values)
    
    @staticmethod
    def calculate_confidence_score(card_data: Dict) -> float:
        """Calculate overall confidence score for extracted card data"""
        scores = []
        
        # Name confidence
        if OCRValidator.validate_card_name(card_data.get('name', '')):
            scores.append(0.3)  # 30% weight for valid name
        
        # Mana cost confidence
        if OCRValidator.validate_mana_cost(card_data.get('mana_cost', '')):
            scores.append(0.2)  # 20% weight for valid mana cost
        
        # Type line confidence
        if card_data.get('type_line'):
            scores.append(0.2)  # 20% weight for having type line
        
        # Oracle text confidence
        if card_data.get('oracle_text') and len(card_data['oracle_text']) > 10:
            scores.append(0.2)  # 20% weight for substantial oracle text
        
        # P/T confidence (if creature)
        type_line = (card_data.get('type_line') or '').lower() if isinstance(card_data.get('type_line'), str) else ''
        if 'creature' in type_line:
            if OCRValidator.validate_power_toughness(card_data.get('power_toughness')):
                scores.append(0.1)  # 10% weight for valid P/T
        else:
            scores.append(0.1)  # 10% for non-creatures (no P/T expected)
        
        return sum(scores)
    
    @staticmethod
    def calculate_confidence_score(card_data: Dict) -> float:
        """Calculate overall confidence score for extracted card data"""
        scores = []
        
        # Name confidence
        if OCRValidator.validate_card_name(card_data.get('name', '')):
            scores.append(0.3)  # 30% weight for valid name
        
        # Mana cost confidence
        if OCRValidator.validate_mana_cost(card_data.get('mana_cost', '')):
            scores.append(0.2)  # 20% weight for valid mana cost
        
        # Type line confidence
        if card_data.get('type_line'):
            scores.append(0.2)  # 20% weight for having type line
        
        # Oracle text confidence
        if card_data.get('oracle_text') and len(card_data['oracle_text']) > 10:
            scores.append(0.2)  # 20% weight for substantial oracle text
        
        # P/T confidence (if creature)
        type_line = (card_data.get('type_line') or '').lower() if isinstance(card_data.get('type_line'), str) else ''
        if 'creature' in type_line:
            if OCRValidator.validate_power_toughness(card_data.get('power_toughness')):
                scores.append(0.1)  # 10% weight for valid P/T
        else:
            scores.append(0.1)  # 10% for non-creatures (no P/T expected)
        
        return sum(scores)
        
        return False
    
    @staticmethod
    def validate_mana_cost(mana_cost: str) -> bool:
        """Validate mana cost format"""
        if not mana_cost:
            return True  # Empty mana cost is valid (for lands, etc.)
        
        # Check for valid mana symbols
        valid_pattern = r'^(\{[WUBRGCTXYS0-9]+(?:/[WUBRG])?\})*$'
        return bool(re.match(valid_pattern, mana_cost))
    
    @staticmethod
    def validate_power_toughness(pt_dict: Dict) -> bool:
        """Validate power/toughness values"""
        if not pt_dict:
            return True  # No P/T is valid for non-creatures
        
        power = pt_dict.get('power')
        toughness = pt_dict.get('toughness')
        
        if not power or not toughness:
            return False
        
        # Check if values are valid (numbers or *)
        valid_values = re.match(r'^(\d+|\*)$', power) and re.match(r'^(\d+|\*)$', toughness)
        return valid_values is not None
        return bool(valid_values)
    
    @staticmethod
    def calculate_confidence_score(card_data: Dict) -> float:
        """Calculate overall confidence score for extracted card data"""
        scores = []
        
        # Name confidence
        if OCRValidator.validate_card_name(card_data.get('name', '')):
            scores.append(0.3)  # 30% weight for valid name
        
        # Mana cost confidence
        if OCRValidator.validate_mana_cost(card_data.get('mana_cost', '')):
            scores.append(0.2)  # 20% weight for valid mana cost
        
        # Type line confidence
        if card_data.get('type_line'):
            scores.append(0.2)  # 20% weight for having type line
        
        # Oracle text confidence
        if card_data.get('oracle_text') and len(card_data['oracle_text']) > 10:
            scores.append(0.2)  # 20% weight for substantial oracle text
        
        # P/T confidence (if creature)
        type_line = (card_data.get('type_line') or '').lower() if isinstance(card_data.get('type_line'), str) else ''
        if 'creature' in type_line:
            if OCRValidator.validate_power_toughness(card_data.get('power_toughness')):
                scores.append(0.1)  # 10% weight for valid P/T
        else:
            scores.append(0.1)  # 10% for non-creatures (no P/T expected)
        
        return sum(scores)
        return bool(valid_values)
    
    @staticmethod
    def calculate_confidence_score(card_data: Dict) -> float:
        """Calculate overall confidence score for extracted card data"""
        scores = []
        
        # Name confidence
        if OCRValidator.validate_card_name(card_data.get('name', '')):
            scores.append(0.3)  # 30% weight for valid name
        
        # Mana cost confidence
        if OCRValidator.validate_mana_cost(card_data.get('mana_cost', '')):
            scores.append(0.2)  # 20% weight for valid mana cost
        
        # Type line confidence
        if card_data.get('type_line'):
            scores.append(0.2)  # 20% weight for having type line
        
        # Oracle text confidence
        if card_data.get('oracle_text') and len(card_data['oracle_text']) > 10:
            scores.append(0.2)  # 20% weight for substantial oracle text
        
        # P/T confidence (if creature)
        type_line = (card_data.get('type_line') or '').lower() if isinstance(card_data.get('type_line'), str) else ''
        if 'creature' in type_line:
            if OCRValidator.validate_power_toughness(card_data.get('power_toughness')):
                scores.append(0.1)  # 10% weight for valid P/T
        else:
            scores.append(0.1)  # 10% for non-creatures (no P/T expected)
        
        return sum(scores)
        
        return False
    
    @staticmethod
    def validate_mana_cost(mana_cost: str) -> bool:
        """Validate mana cost format"""
        if not mana_cost:
            return True  # Empty mana cost is valid (for lands, etc.)
        
        # Check for valid mana symbols
        valid_pattern = r'^(\{[WUBRGCTXYS0-9]+(?:/[WUBRG])?\})*$'
        return bool(re.match(valid_pattern, mana_cost))
    
    @staticmethod
    def validate_power_toughness(pt_dict: Dict) -> bool:
        """Validate power/toughness values"""
        if not pt_dict:
            return True  # No P/T is valid for non-creatures
        
        power = pt_dict.get('power')
        toughness = pt_dict.get('toughness')
        
        if not power or not toughness:
            return False
        
        # Check if values are valid (numbers or *)
        valid_values = re.match(r'^(\d+|\*)$', power) and re.match(r'^(\d+|\*)$', toughness)
        return valid_values is not None
        return bool(valid_values)
    
    @staticmethod
    def calculate_confidence_score(card_data: Dict) -> float:
        """Calculate overall confidence score for extracted card data"""
        scores = []
        
        # Name confidence
        if OCRValidator.validate_card_name(card_data.get('name', '')):
            scores.append(0.3)  # 30% weight for valid name
        
        # Mana cost confidence
        if OCRValidator.validate_mana_cost(card_data.get('mana_cost', '')):
            scores.append(0.2)  # 20% weight for valid mana cost
        
        # Type line confidence
        if card_data.get('type_line'):
            scores.append(0.2)  # 20% weight for having type line
        
        # Oracle text confidence
        if card_data.get('oracle_text') and len(card_data['oracle_text']) > 10:
            scores.append(0.2)  # 20% weight for substantial oracle text
        
        # P/T confidence (if creature)
        type_line = (card_data.get('type_line') or '').lower() if isinstance(card_data.get('type_line'), str) else ''
        if 'creature' in type_line:
            if OCRValidator.validate_power_toughness(card_data.get('power_toughness')):
                scores.append(0.1)  # 10% weight for valid P/T
        else:
            scores.append(0.1)  # 10% for non-creatures (no P/T expected)
        
        return sum(scores)
        
        if text_blocks:
            # Sort by Y position and combine
            text_blocks.sort(key=lambda x: x[1])
            oracle_text = ' '.join([block[0] for block in text_blocks])
            
            # Clean up the text
            oracle_text = self._clean_oracle_text(oracle_text)
            return oracle_text if len(oracle_text) > 10 else None
        
        return None
    
    def _extract_flavor_text(self, ocr_results: List[Dict], image_shape: Tuple) -> Optional[str]:
        """Extract flavor text (usually italicized) from lower region"""
        height = image_shape[0]
        flavor_candidates = []
        
        for result in ocr_results:
            bbox = result['bbox']
            text = result['text']
            confidence = result['confidence']
            
            # Calculate center Y coordinate
            y_center = sum(point[1] for point in bbox) / 4
            
            # Flavor text is typically in the lower region (70%-90% from top)
            if 0.7 * height < y_center < 0.9 * height and confidence > 0.5:
                # Flavor text often contains quotes or is more descriptive
                if ('"' in text or len(text.split()) > 5) and len(text) > 10:
                    flavor_candidates.append((text, y_center))
        
        if flavor_candidates:
            # Combine all potential flavor text
            flavor_candidates.sort(key=lambda x: x[1])
            flavor_text = ' '.join([candidate[0] for candidate in flavor_candidates])
            return flavor_text.strip()
        
        return None
    
    def _extract_power_toughness(self, ocr_results: List[Dict], image_shape: Tuple) -> Optional[Dict]:
        """Extract power/toughness from bottom-right region"""
        height, width = image_shape[:2]
        pt_candidates = []
        
        for result in ocr_results:
            bbox = result['bbox']
            text = result['text']
            confidence = result['confidence']
            
            # Calculate center coordinates
            x_center = sum(point[0] for point in bbox) / 4
            y_center = sum(point[1] for point in bbox) / 4
            
            # P/T is typically in bottom-right corner
            if (x_center > width * 0.7 and y_center > height * 0.8 and 
                confidence > 0.6):
                
                # Look for P/T pattern
                pt_match = re.search(self.pt_pattern, text)
                if pt_match:
                    power, toughness = pt_match.groups()
                    return {
                        'power': power,
                        'toughness': toughness,
                        'combined': f"{power}/{toughness}"
                    }
                
                # Sometimes OCR splits P/T across multiple detections
                pt_candidates.append((text, x_center, y_center))
        
        # Try to reconstruct P/T from nearby text
        if pt_candidates and len(pt_candidates) >= 2:
            # Sort by position and try to find P/T pattern
            pt_candidates.sort(key=lambda x: (x[2], x[1]))  # Sort by Y then X
            combined_text = ''.join([candidate[0] for candidate in pt_candidates[-3:]])
            
            pt_match = re.search(self.pt_pattern, combined_text)
            if pt_match:
                power, toughness = pt_match.groups()
                return {
                    'power': power,
                    'toughness': toughness,
                    'combined': f"{power}/{toughness}"
                }
        
        return None
    
    def _extract_rarity(self, ocr_results: List[Dict]) -> Optional[str]:
        """Extract rarity information"""
        all_text = ' '.join([result['text'].lower() for result in ocr_results if result.get('text')])
        
        rarity_keywords = {
            'common': ['common', 'c'],
            'uncommon': ['uncommon', 'u'],
            'rare': ['rare', 'r'],
            'mythic': ['mythic', 'mythic rare', 'm']
        }
        
        for rarity, keywords in rarity_keywords.items():
            for keyword in keywords:
                if keyword in all_text:
                    return rarity
        
        return None
    
    def _clean_oracle_text(self, text: str) -> str:
        """Clean and normalize oracle text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR errors in MTG text
        replacements = {
            r'\{T\}': '{T}',  # Tap symbol
            r'\{([WUBRG])\}': r'{\1}',  # Mana symbols
            r'(\d+)/(\d+)': r'\1/\2',  # Power/toughness format
            r'\benter the battlefield\b': 'enters the battlefield',
            r'\bgraveyard\b': 'graveyard',
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _calculate_overall_confidence(self, ocr_results: List[Dict]) -> float:
        """Calculate overall confidence score for the extraction"""
        if not ocr_results:
            return 0.0
        
        confidences = [result['confidence'] for result in ocr_results]
        
        # Weight by text length (longer text blocks are more reliable)
        weighted_confidence = 0
        total_weight = 0
        
        for result in ocr_results:
            weight = len(result['text'])
            weighted_confidence += result['confidence'] * weight
            total_weight += weight
        
        if total_weight > 0:
            return weighted_confidence / total_weight
        else:
            return sum(confidences) / len(confidences)
    
    def _post_process_card_data(self, card_data: Dict) -> Dict:
        """Post-process and validate extracted card data"""
        # Clean card name
        if card_data.get('name'):
            card_data['name'] = self._clean_card_name(card_data['name'])
        
        # Validate and normalize mana cost
        if card_data.get('mana_cost'):
            card_data['converted_mana_cost'] = self._calculate_cmc(card_data['mana_cost'])
        
        # Extract colors from mana cost
        if card_data.get('mana_cost'):
            card_data['colors'] = self._extract_colors(card_data['mana_cost'])
        
        # Validate P/T for creatures
        if card_data.get('type_line') and isinstance(card_data['type_line'], str) and 'creature' in card_data['type_line'].lower():
            if not card_data.get('power_toughness'):
                # Try to extract P/T from oracle text as fallback
                pt_from_text = self._extract_pt_from_text(card_data.get('oracle_text', ''))
                if pt_from_text:
                    card_data['power_toughness'] = pt_from_text
        
        return card_data
    
    def _clean_card_name(self, name: str) -> str:
        """Clean and normalize card name"""
        # Remove common OCR artifacts
        name = re.sub(r'[^\w\s\-\',]', '', name)
        name = re.sub(r'\s+', ' ', name)
        return name.strip()
    
    def _calculate_cmc(self, mana_cost: str) -> int:
        """Calculate converted mana cost"""
        if not mana_cost:
            return 0
        
        cmc = 0
        # Extract numbers from mana cost
        numbers = re.findall(r'\{(\d+)\}', mana_cost)
        for num in numbers:
            cmc += int(num)
        
        # Count colored mana symbols
        colored_symbols = re.findall(r'\{([WUBRG])\}', mana_cost)
        cmc += len(colored_symbols)
        
        # Count hybrid mana (counts as 1 each)
        hybrid_symbols = re.findall(r'\{[WUBRG]/[WUBRG]\}', mana_cost)
        cmc += len(hybrid_symbols)
        
        return cmc
    
    def _extract_colors(self, mana_cost: str) -> str:
        """Extract color identity from mana cost"""
        if not mana_cost:
            return ''
        
        colors = set()
        
        # Find colored mana symbols
        colored_symbols = re.findall(r'\{([WUBRG])\}', mana_cost)
        colors.update(colored_symbols)
        
        # Find hybrid mana symbols
        hybrid_symbols = re.findall(r'\{([WUBRG])/([WUBRG])\}', mana_cost)
        for symbol in hybrid_symbols:
            colors.update(symbol)
        
        # Sort colors in WUBRG order
        color_order = 'WUBRG'
        sorted_colors = ''.join(sorted(colors, key=lambda x: color_order.index(x)))
        
        return sorted_colors
    
    def _extract_pt_from_text(self, text: str) -> Optional[Dict]:
        """Try to extract P/T from oracle text as fallback"""
        if not text:
            return None
        
        # Look for P/T patterns in text
        pt_match = re.search(self.pt_pattern, text)
        if pt_match:
            power, toughness = pt_match.groups()
            return {
                'power': power,
                'toughness': toughness,
                'combined': f"{power}/{toughness}"
            }
        
        return None

class OCRValidator:
    """Validates OCR results against MTG card conventions"""
    
    @staticmethod
    def validate_card_name(name: str) -> bool:
        """Validate that extracted name looks like a real card name"""
        if not name or len(name) < 2:
            return False
        
        # Check for reasonable character composition
        if re.match(r'^[A-Za-z\s\-\',]+$', name):
            return True
        
        return False
    
    @staticmethod
    def validate_mana_cost(mana_cost: str) -> bool:
        """Validate mana cost format"""
        if not mana_cost:
            return True  # Empty mana cost is valid (for lands, etc.)
        
        # Check for valid mana symbols
        valid_pattern = r'^(\{[WUBRGCTXYS0-9]+(?:/[WUBRG])?\})*$'
        return bool(re.match(valid_pattern, mana_cost))
    
    @staticmethod
    def validate_power_toughness(pt_dict: Dict) -> bool:
        """Validate power/toughness values"""
        if not pt_dict:
            return True  # No P/T is valid for non-creatures
        
        power = pt_dict.get('power')
        toughness = pt_dict.get('toughness')
        
        if not power or not toughness:
            return False
        
        # Check if values are valid (numbers or *)
        valid_values = re.match(r'^(\d+|\*)$', power) and re.match(r'^(\d+|\*)$', toughness)
        return valid_values is not None
    
    @staticmethod
    def calculate_confidence_score(card_data: Dict) -> float:
        """Calculate overall confidence score for extracted card data"""
        scores = []
        
        # Name confidence
        if OCRValidator.validate_card_name(card_data.get('name', '')):
            scores.append(0.3)  # 30% weight for valid name
        
        # Mana cost confidence
        if OCRValidator.validate_mana_cost(card_data.get('mana_cost', '')):
            scores.append(0.2)  # 20% weight for valid mana cost
        
        # Type line confidence
        if card_data.get('type_line'):
            scores.append(0.2)  # 20% weight for having type line
        
        # Oracle text confidence
        if card_data.get('oracle_text') and len(card_data['oracle_text']) > 10:
            scores.append(0.2)  # 20% weight for substantial oracle text
        
        # P/T confidence (if creature)
        type_line = (card_data.get('type_line') or '').lower() if isinstance(card_data.get('type_line'), str) else ''
        if 'creature' in type_line:
            if OCRValidator.validate_power_toughness(card_data.get('power_toughness')):
                scores.append(0.1)  # 10% weight for valid P/T
        else:
            scores.append(0.1)  # 10% for non-creatures (no P/T expected)
        
        return sum(scores)
        
        return False
    
    @staticmethod
    def validate_mana_cost(mana_cost: str) -> bool:
        """Validate mana cost format"""
        if not mana_cost:
            return True  # Empty mana cost is valid (for lands, etc.)
        
        # Check for valid mana symbols
        valid_pattern = r'^(\{[WUBRGCTXYS0-9]+(?:/[WUBRG])?\})*$'
        return bool(re.match(valid_pattern, mana_cost))
    
    @staticmethod
    def validate_power_toughness(pt_dict: Dict) -> bool:
        """Validate power/toughness values"""
        if not pt_dict:
            return True  # No P/T is valid for non-creatures
        
        power = pt_dict.get('power')
        toughness = pt_dict.get('toughness')
        
        if not power or not toughness:
            return False
        
        # Check if values are valid (numbers or *)
        valid_values = re.match(r'^(\d+|\*)$', power) and re.match(r'^(\d+|\*)$', toughness)
        return valid_values is not None
        return bool(valid_values)
    
    @staticmethod
    def calculate_confidence_score(card_data: Dict) -> float:
        """Calculate overall confidence score for extracted card data"""
        scores = []
        
        # Name confidence
        if OCRValidator.validate_card_name(card_data.get('name', '')):
            scores.append(0.3)  # 30% weight for valid name
        
        # Mana cost confidence
        if OCRValidator.validate_mana_cost(card_data.get('mana_cost', '')):
            scores.append(0.2)  # 20% weight for valid mana cost
        
        # Type line confidence
        if card_data.get('type_line'):
            scores.append(0.2)  # 20% weight for having type line
        
        # Oracle text confidence
        if card_data.get('oracle_text') and len(card_data['oracle_text']) > 10:
            scores.append(0.2)  # 20% weight for substantial oracle text
        
        # P/T confidence (if creature)
        type_line = (card_data.get('type_line') or '').lower() if isinstance(card_data.get('type_line'), str) else ''
        if 'creature' in type_line:
            if OCRValidator.validate_power_toughness(card_data.get('power_toughness')):
                scores.append(0.1)  # 10% weight for valid P/T
        else:
            scores.append(0.1)  # 10% for non-creatures (no P/T expected)
        
        return sum(scores)
    
    @staticmethod
    def calculate_confidence_score(card_data: Dict) -> float:
        """Calculate overall confidence score for extracted card data"""
        scores = []
        
        # Name confidence
        if OCRValidator.validate_card_name(card_data.get('name', '')):
            scores.append(0.3)  # 30% weight for valid name
        
        # Mana cost confidence
        if OCRValidator.validate_mana_cost(card_data.get('mana_cost', '')):
            scores.append(0.2)  # 20% weight for valid mana cost
        
        # Type line confidence
        if card_data.get('type_line'):
            scores.append(0.2)  # 20% weight for having type line
        
        # Oracle text confidence
        if card_data.get('oracle_text') and len(card_data['oracle_text']) > 10:
            scores.append(0.2)  # 20% weight for substantial oracle text
        
        # P/T confidence (if creature)
        type_line = (card_data.get('type_line') or '').lower() if isinstance(card_data.get('type_line'), str) else ''
        if 'creature' in type_line:
            if OCRValidator.validate_power_toughness(card_data.get('power_toughness')):
                scores.append(0.1)  # 10% weight for valid P/T
        else:
            scores.append(0.1)  # 10% for non-creatures (no P/T expected)
        
        return sum(scores)
        
        return False
    
    @staticmethod
    def validate_mana_cost(mana_cost: str) -> bool:
        """Validate mana cost format"""
        if not mana_cost:
            return True  # Empty mana cost is valid (for lands, etc.)
        
        # Check for valid mana symbols
        valid_pattern = r'^(\{[WUBRGCTXYS0-9]+(?:/[WUBRG])?\})*$'
        return bool(re.match(valid_pattern, mana_cost))
    
    @staticmethod
    def validate_power_toughness(pt_dict: Dict) -> bool:
        """Validate power/toughness values"""
        if not pt_dict:
            return True  # No P/T is valid for non-creatures
        
        power = pt_dict.get('power')
        toughness = pt_dict.get('toughness')
        
        if not power or not toughness:
            return False
        
        # Check if values are valid (numbers or *)
        valid_values = re.match(r'^(\d+|\*)$', power) and re.match(r'^(\d+|\*)$', toughness)
        return valid_values is not None
        return bool(valid_values)
    
    @staticmethod
    def calculate_confidence_score(card_data: Dict) -> float:
        """Calculate overall confidence score for extracted card data"""
        scores = []
        
        # Name confidence
        if OCRValidator.validate_card_name(card_data.get('name', '')):
            scores.append(0.3)  # 30% weight for valid name
        
        # Mana cost confidence
        if OCRValidator.validate_mana_cost(card_data.get('mana_cost', '')):
            scores.append(0.2)  # 20% weight for valid mana cost
        
        # Type line confidence
        if card_data.get('type_line'):
            scores.append(0.2)  # 20% weight for having type line
        
        # Oracle text confidence
        if card_data.get('oracle_text') and len(card_data['oracle_text']) > 10:
            scores.append(0.2)  # 20% weight for substantial oracle text
        
        # P/T confidence (if creature)
        type_line = (card_data.get('type_line') or '').lower() if isinstance(card_data.get('type_line'), str) else ''
        if 'creature' in type_line:
            if OCRValidator.validate_power_toughness(card_data.get('power_toughness')):
                scores.append(0.1)  # 10% weight for valid P/T
        else:
            scores.append(0.1)  # 10% for non-creatures (no P/T expected)
        
        return sum(scores)
        return bool(valid_values)
    
    @staticmethod
    def calculate_confidence_score(card_data: Dict) -> float:
        """Calculate overall confidence score for extracted card data"""
        scores = []
        
        # Name confidence
        if OCRValidator.validate_card_name(card_data.get('name', '')):
            scores.append(0.3)  # 30% weight for valid name
        
        # Mana cost confidence
        if OCRValidator.validate_mana_cost(card_data.get('mana_cost', '')):
            scores.append(0.2)  # 20% weight for valid mana cost
        
        # Type line confidence
        if card_data.get('type_line'):
            scores.append(0.2)  # 20% weight for having type line
        
        # Oracle text confidence
        if card_data.get('oracle_text') and len(card_data['oracle_text']) > 10:
            scores.append(0.2)  # 20% weight for substantial oracle text
        
        # P/T confidence (if creature)
        type_line = (card_data.get('type_line') or '').lower() if isinstance(card_data.get('type_line'), str) else ''
        if 'creature' in type_line:
            if OCRValidator.validate_power_toughness(card_data.get('power_toughness')):
                scores.append(0.1)  # 10% weight for valid P/T
        else:
            scores.append(0.1)  # 10% for non-creatures (no P/T expected)
        
        return sum(scores)
        
        return False
    
    @staticmethod
    def validate_mana_cost(mana_cost: str) -> bool:
        """Validate mana cost format"""
        if not mana_cost:
            return True  # Empty mana cost is valid (for lands, etc.)
        
        # Check for valid mana symbols
        valid_pattern = r'^(\{[WUBRGCTXYS0-9]+(?:/[WUBRG])?\})*$'
        return bool(re.match(valid_pattern, mana_cost))
    
    @staticmethod
    def validate_power_toughness(pt_dict: Dict) -> bool:
        """Validate power/toughness values"""
        if not pt_dict:
            return True  # No P/T is valid for non-creatures
        
        power = pt_dict.get('power')
        toughness = pt_dict.get('toughness')
        
        if not power or not toughness:
            return False
        
        # Check if values are valid (numbers or *)
        valid_values = re.match(r'^(\d+|\*)$', power) and re.match(r'^(\d+|\*)$', toughness)
        return valid_values is not None
        return bool(valid_values)
    
    @staticmethod
    def calculate_confidence_score(card_data: Dict) -> float:
        """Calculate overall confidence score for extracted card data"""
        scores = []
        
        # Name confidence
        if OCRValidator.validate_card_name(card_data.get('name', '')):
            scores.append(0.3)  # 30% weight for valid name
        
        # Mana cost confidence
        if OCRValidator.validate_mana_cost(card_data.get('mana_cost', '')):
            scores.append(0.2)  # 20% weight for valid mana cost
        
        # Type line confidence
        if card_data.get('type_line'):
            scores.append(0.2)  # 20% weight for having type line
        
        # Oracle text confidence
        if card_data.get('oracle_text') and len(card_data['oracle_text']) > 10:
            scores.append(0.2)  # 20% weight for substantial oracle text
        
        # P/T confidence (if creature)
        type_line = (card_data.get('type_line') or '').lower() if isinstance(card_data.get('type_line'), str) else ''
        if 'creature' in type_line:
            if OCRValidator.validate_power_toughness(card_data.get('power_toughness')):
                scores.append(0.1)  # 10% weight for valid P/T
        else:
            scores.append(0.1)  # 10% for non-creatures (no P/T expected)
        
        return sum(scores)