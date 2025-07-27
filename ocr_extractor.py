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
    logging.warning("Tesseract not available")

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
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
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
        
        # Common MTG keywords for validation
        self.mtg_keywords = {
            'flying', 'trample', 'haste', 'vigilance', 'deathtouch', 'lifelink',
            'first strike', 'double strike', 'hexproof', 'shroud', 'defender',
            'reach', 'flash', 'prowess', 'menace', 'indestructible'
        }
    
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
                return {}
            
            # Try targeted extraction first for better results
            card_name = self._extract_card_name_targeted(card_image)
            
            # If targeted extraction fails, fall back to position-based extraction
            if not card_name or len(card_name) < 3:
                card_name = self._extract_card_name(ocr_results, card_image.shape)
                logger.debug("Using position-based card name extraction as fallback")
            
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
            
            logger.debug(f"Extracted card data: {card_data['name']} - Confidence: {card_data['confidence']:.2f}")
            return card_data
            
        except Exception as e:
            logger.error(f"Error extracting card attributes: {str(e)}")
            return {}
    
    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Optimize image for OCR accuracy with enhanced preprocessing"""
        try:
            # Resize very small images
            height, width = image.shape[:2]
            if height < 200 or width < 200:
                # Scale up small images
                scale_factor = max(200 / height, 200 / width)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            
            # Convert to PIL for enhancement
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            # Enhance contrast more aggressively
            enhancer = ImageEnhance.Contrast(pil_image)
            enhanced = enhancer.enhance(2.0)  # Increased from 1.3
            
            # Enhance sharpness more aggressively
            enhancer = ImageEnhance.Sharpness(enhanced)
            sharpened = enhancer.enhance(2.0)  # Increased from 1.2
            
            # Enhance brightness slightly
            enhancer = ImageEnhance.Brightness(sharpened)
            brightened = enhancer.enhance(1.1)
            
            # Convert back to OpenCV format
            if len(image.shape) == 3:
                result = cv2.cvtColor(np.array(brightened), cv2.COLOR_RGB2BGR)
            else:
                result = np.array(brightened)
            
            # Apply additional OpenCV preprocessing
            if len(result.shape) == 3:
                gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            else:
                gray = result
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
            
            # Apply denoising
            denoised = cv2.fastNlMeansDenoising(enhanced_gray, None, 10, 7, 21)
            
            # Convert back to color if original was color
            if len(image.shape) == 3:
                result = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
            else:
                result = denoised
            
            return result
            
        except Exception as e:
            logger.warning(f"Enhanced image preprocessing failed: {str(e)}")
            return image
    
    def _perform_ocr(self, image: np.ndarray) -> List[Dict]:
        """Perform OCR using the configured engine"""
        results = []
        
        try:
            if self.ocr_engine == 'easyocr' and self.easyocr_reader:
                results = self._easyocr_extract(image)
            elif self.ocr_engine == 'tesseract' and TESSERACT_AVAILABLE:
                results = self._tesseract_extract(image)
            elif self.ocr_engine == 'both':
                # Try EasyOCR first, fallback to Tesseract
                if self.easyocr_reader:
                    results = self._easyocr_extract(image)
                if not results and TESSERACT_AVAILABLE:
                    results = self._tesseract_extract(image)
            else:
                logger.error(f"OCR engine '{self.ocr_engine}' not available")
        
        except Exception as e:
            logger.error(f"OCR extraction failed: {str(e)}")
        
        return results
    
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
            logger.error(f"Tesseract extraction failed: {str(e)}")
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
            
            # Card name is in the top banner - very top 15% of card
            if y_center < height * 0.15 and confidence > 0.6:
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
        return None
    
    def _extract_card_name_targeted(self, card_image: np.ndarray) -> Optional[str]:
        """Extract card name using targeted OCR on title banner region only"""
        try:
            height, width = card_image.shape[:2]
            
            # Crop the title banner region (top 12% of card)
            title_region = card_image[:int(height * 0.12), :]
            
            if title_region.size == 0:
                return None
            
            # Enhance the title region specifically for text recognition
            enhanced_title = self._preprocess_title_region(title_region)
            
            # Perform OCR on just the title region
            title_ocr_results = self._perform_ocr(enhanced_title)
            
            if not title_ocr_results:
                return None
            
            # Find the best text candidate from title region
            best_text = None
            best_score = 0
            
            for result in title_ocr_results:
                text = result['text'].strip()
                confidence = result['confidence']
                
                # Filter out obvious non-name text
                if (len(text) >= 3 and 
                    confidence > 0.7 and
                    not re.match(r'^[\d\{\}/\*\(\)]+$', text) and
                    not re.search(r'\billus\b|©|\bcopyright\b', text, re.IGNORECASE) and
                    not text.isnumeric()):
                    
                    if confidence > best_score:
                        best_score = confidence
                        best_text = text
            
            if best_text:
                logger.debug(f"Targeted card name extraction: '{best_text}' (confidence: {best_score:.2f})")
                return best_text
                
        except Exception as e:
            logger.debug(f"Targeted card name extraction failed: {str(e)}")
        
        return None
    
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
        if card_data.get('type_line') and 'creature' in card_data['type_line'].lower():
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
        type_line = card_data.get('type_line', '').lower()
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
        type_line = card_data.get('type_line', '').lower()
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
        type_line = card_data.get('type_line', '').lower()
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
        type_line = card_data.get('type_line', '').lower()
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
        type_line = card_data.get('type_line', '').lower()
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
        type_line = card_data.get('type_line', '').lower()
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
        if card_data.get('type_line') and 'creature' in card_data['type_line'].lower():
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
        type_line = card_data.get('type_line', '').lower()
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
        type_line = card_data.get('type_line', '').lower()
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
        type_line = card_data.get('type_line', '').lower()
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
        type_line = card_data.get('type_line', '').lower()
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
        type_line = card_data.get('type_line', '').lower()
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
        type_line = card_data.get('type_line', '').lower()
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
        if card_data.get('type_line') and 'creature' in card_data['type_line'].lower():
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
        type_line = card_data.get('type_line', '').lower()
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
        type_line = card_data.get('type_line', '').lower()
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
        type_line = card_data.get('type_line', '').lower()
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
        type_line = card_data.get('type_line', '').lower()
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
        type_line = card_data.get('type_line', '').lower()
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
        type_line = card_data.get('type_line', '').lower()
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
        if card_data.get('type_line') and 'creature' in card_data['type_line'].lower():
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
        type_line = card_data.get('type_line', '').lower()
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
        type_line = card_data.get('type_line', '').lower()
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
        type_line = card_data.get('type_line', '').lower()
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
        type_line = card_data.get('type_line', '').lower()
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
        type_line = card_data.get('type_line', '').lower()
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
        type_line = card_data.get('type_line', '').lower()
        if 'creature' in type_line:
            if OCRValidator.validate_power_toughness(card_data.get('power_toughness')):
                scores.append(0.1)  # 10% weight for valid P/T
        else:
            scores.append(0.1)  # 10% for non-creatures (no P/T expected)
        
        return sum(scores)