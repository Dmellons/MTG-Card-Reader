"""
Card Detection and Segmentation Module
Detects and extracts individual MTG cards from PDF pages containing multiple cards
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict
import logging
from config import settings

logger = logging.getLogger(__name__)

class CardDetector:
    """Detects and segments individual cards from grid layouts"""
    
    def __init__(self):
        self.expected_rows = settings.EXPECTED_GRID_ROWS
        self.expected_cols = settings.EXPECTED_GRID_COLS
        self.perspective_correction = settings.PERSPECTIVE_CORRECTION
    
    def detect_card_grid_advanced(self, image: np.ndarray, 
                                 expected_rows: int = None, 
                                 expected_cols: int = None) -> List[np.ndarray]:
        """
        Robust grid detection using multiple CV techniques
        Returns list of individual card images
        """
        if expected_rows is None:
            expected_rows = self.expected_rows
        if expected_cols is None:
            expected_cols = self.expected_cols
        
        try:
            logger.debug(f"Detecting {expected_rows}x{expected_cols} card grid")
            
            # Preprocess image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Enhanced preprocessing with CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Multi-scale edge detection
            edges = cv2.Canny(enhanced, 50, 150, apertureSize=3)
            
            # Apply morphological operations to connect broken lines
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Detect lines using Probabilistic Hough Transform
            lines = cv2.HoughLinesP(
                edges, 1, np.pi/180, 
                threshold=100, 
                minLineLength=min(image.shape[:2]) // 10, 
                maxLineGap=20
            )
            
            if lines is None:
                logger.warning("No lines detected, falling back to contour detection")
                return self._detect_cards_by_contours(image, expected_rows, expected_cols)
            
            # Classify and filter lines
            h_lines, v_lines = self._classify_grid_lines(lines)
            
            if len(h_lines) < 2 or len(v_lines) < 2:
                logger.warning("Insufficient grid lines detected, falling back to contour detection")
                return self._detect_cards_by_contours(image, expected_rows, expected_cols)
            
            # Extract card regions
            cards = self._extract_card_regions_robust(image, h_lines, v_lines, 
                                                     expected_rows, expected_cols)
            
            logger.info(f"Successfully detected {len(cards)} cards using grid detection")
            return cards
            
        except Exception as e:
            logger.error(f"Grid detection failed: {str(e)}")
            return self._detect_cards_by_contours(image, expected_rows, expected_cols)
    
    def _classify_grid_lines(self, lines: np.ndarray) -> Tuple[List, List]:
        """Separate horizontal and vertical lines with tolerance"""
        h_lines, v_lines = [], []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle
            if x2 - x1 == 0:  # Vertical line
                angle = 90
            else:
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            # Classify with tolerance
            if abs(angle) < 15 or abs(angle - 180) < 15 or abs(angle + 180) < 15:
                h_lines.append(line)
            elif abs(angle - 90) < 15 or abs(angle + 90) < 15:
                v_lines.append(line)
        
        # Remove duplicate lines
        h_lines = self._merge_similar_lines(h_lines, is_horizontal=True)
        v_lines = self._merge_similar_lines(v_lines, is_horizontal=False)
        
        return h_lines, v_lines
    
    def _merge_similar_lines(self, lines: List, is_horizontal: bool, tolerance: int = 20) -> List:
        """Merge lines that are close to each other"""
        if not lines:
            return []
        
        merged_lines = []
        used = set()
        
        for i, line1 in enumerate(lines):
            if i in used:
                continue
            
            x1, y1, x2, y2 = line1[0]
            similar_lines = [line1]
            used.add(i)
            
            for j, line2 in enumerate(lines[i+1:], i+1):
                if j in used:
                    continue
                
                x3, y3, x4, y4 = line2[0]
                
                # Check if lines are parallel and close
                if is_horizontal:
                    # For horizontal lines, check y-coordinate distance
                    avg_y1 = (y1 + y2) / 2
                    avg_y2 = (y3 + y4) / 2
                    if abs(avg_y1 - avg_y2) < tolerance:
                        similar_lines.append(line2)
                        used.add(j)
                else:
                    # For vertical lines, check x-coordinate distance
                    avg_x1 = (x1 + x2) / 2
                    avg_x2 = (x3 + x4) / 2
                    if abs(avg_x1 - avg_x2) < tolerance:
                        similar_lines.append(line2)
                        used.add(j)
            
            # Create averaged line from similar lines
            if similar_lines:
                merged_line = self._average_lines(similar_lines)
                merged_lines.append(merged_line)
        
        return merged_lines
    
    def _average_lines(self, lines: List) -> np.ndarray:
        """Average multiple similar lines into one"""
        total_x1, total_y1, total_x2, total_y2 = 0, 0, 0, 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            total_x1 += x1
            total_y1 += y1
            total_x2 += x2
            total_y2 += y2
        
        count = len(lines)
        avg_line = np.array([[
            total_x1 // count, total_y1 // count,
            total_x2 // count, total_y2 // count
        ]])
        
        return avg_line
    
    def _extract_card_regions_robust(self, image: np.ndarray, h_lines: List, 
                                   v_lines: List, rows: int, cols: int) -> List[np.ndarray]:
        """Extract card regions from detected grid lines"""
        try:
            height, width = image.shape[:2]
            
            # Sort lines by position
            h_positions = []
            for line in h_lines:
                x1, y1, x2, y2 = line[0]
                avg_y = (y1 + y2) / 2
                h_positions.append(avg_y)
            h_positions.sort()
            
            v_positions = []
            for line in v_lines:
                x1, y1, x2, y2 = line[0]
                avg_x = (x1 + x2) / 2
                v_positions.append(avg_x)
            v_positions.sort()
            
            # Add image boundaries if missing
            if h_positions[0] > height * 0.1:
                h_positions.insert(0, 0)
            if h_positions[-1] < height * 0.9:
                h_positions.append(height)
            
            if v_positions[0] > width * 0.1:
                v_positions.insert(0, 0)
            if v_positions[-1] < width * 0.9:
                v_positions.append(width)
            
            # Extract card regions
            cards = []
            for i in range(len(h_positions) - 1):
                for j in range(len(v_positions) - 1):
                    y1 = int(h_positions[i])
                    y2 = int(h_positions[i + 1])
                    x1 = int(v_positions[j])
                    x2 = int(v_positions[j + 1])
                    
                    # Add small padding and ensure bounds
                    padding = 5
                    y1 = max(0, y1 + padding)
                    y2 = min(height, y2 - padding)
                    x1 = max(0, x1 + padding)
                    x2 = min(width, x2 - padding)
                    
                    if y2 > y1 and x2 > x1:
                        card_region = image[y1:y2, x1:x2]
                        if self._is_valid_card_region(card_region):
                            cards.append(card_region)
            
            return cards
            
        except Exception as e:
            logger.error(f"Error extracting card regions: {str(e)}")
            return []
    
    def _detect_cards_by_contours(self, image: np.ndarray, 
                                 expected_rows: int, expected_cols: int) -> List[np.ndarray]:
        """Alternative card detection using contour detection"""
        try:
            logger.debug("Using contour-based card detection")
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Adaptive threshold
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area and aspect ratio
            valid_contours = []
            min_area = (image.shape[0] * image.shape[1]) // (expected_rows * expected_cols * 4)
            max_area = (image.shape[0] * image.shape[1]) // (expected_rows * expected_cols // 2)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    # Check aspect ratio (cards should be roughly rectangular)
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    if 0.6 < aspect_ratio < 1.8:  # Reasonable aspect ratio for cards
                        valid_contours.append(contour)
            
            # Sort by area and take the largest ones
            valid_contours.sort(key=cv2.contourArea, reverse=True)
            target_count = expected_rows * expected_cols
            valid_contours = valid_contours[:target_count]
            
            # Extract card regions
            cards = []
            for contour in valid_contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Add small padding
                padding = 5
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2 * padding)
                h = min(image.shape[0] - y, h + 2 * padding)
                
                card_region = image[y:y+h, x:x+w]
                if self._is_valid_card_region(card_region):
                    cards.append(card_region)
            
            logger.info(f"Detected {len(cards)} cards using contour detection")
            return cards
            
        except Exception as e:
            logger.error(f"Contour detection failed: {str(e)}")
            return []
    
    def _is_valid_card_region(self, card_region: np.ndarray) -> bool:
        """Validate that a detected region looks like a card"""
        if card_region is None or card_region.size == 0:
            return False
        
        height, width = card_region.shape[:2]
        
        # Check minimum size
        if height < 100 or width < 100:
            return False
        
        # Check aspect ratio (MTG cards are roughly 2.5:3.5 ratio)
        aspect_ratio = float(width) / height
        if not (0.5 < aspect_ratio < 2.0):
            return False
        
        # Check that the region has sufficient content (not mostly white/black)
        gray = cv2.cvtColor(card_region, cv2.COLOR_BGR2GRAY) if len(card_region.shape) == 3 else card_region
        
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        total_pixels = height * width
        
        # Check if too much white (background) or black
        white_ratio = hist[250:].sum() / total_pixels
        black_ratio = hist[:5].sum() / total_pixels
        
        if white_ratio > 0.8 or black_ratio > 0.8:
            return False
        
        return True
    
    def detect_cards_template_matching(self, image: np.ndarray, 
                                     template_path: str = None) -> List[np.ndarray]:
        """
        Alternative detection method using template matching
        Useful when you have a template of a card
        """
        try:
            if template_path is None:
                logger.warning("No template provided for template matching")
                return []
            
            # Load template
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                logger.error(f"Could not load template from {template_path}")
                return []
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Perform template matching
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            
            # Find matches above threshold
            threshold = 0.7
            locations = np.where(result >= threshold)
            
            # Extract card regions
            cards = []
            h, w = template.shape
            
            for pt in zip(*locations[::-1]):
                x, y = pt
                card_region = image[y:y+h, x:x+w]
                if self._is_valid_card_region(card_region):
                    cards.append(card_region)
            
            # Remove overlapping detections
            cards = self._remove_overlapping_detections(cards, overlap_threshold=0.5)
            
            logger.info(f"Template matching detected {len(cards)} cards")
            return cards
            
        except Exception as e:
            logger.error(f"Template matching failed: {str(e)}")
            return []
    
    def _remove_overlapping_detections(self, cards: List[np.ndarray], 
                                     overlap_threshold: float = 0.5) -> List[np.ndarray]:
        """Remove overlapping card detections using Non-Maximum Suppression"""
        if len(cards) <= 1:
            return cards
        
        # This is a simplified version - in practice, you'd want to implement
        # proper Non-Maximum Suppression with bounding boxes
        unique_cards = []
        
        for i, card1 in enumerate(cards):
            is_duplicate = False
            
            for j, card2 in enumerate(unique_cards):
                # Simple size-based duplicate detection
                if abs(card1.shape[0] - card2.shape[0]) < 20 and abs(card1.shape[1] - card2.shape[1]) < 20:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_cards.append(card1)
        
        return unique_cards

class GridLayoutAnalyzer:
    """Analyzes page layout to determine optimal grid detection parameters"""
    
    @staticmethod
    def analyze_page_layout(image: np.ndarray) -> Dict:
        """
        Analyze page layout to determine likely grid structure
        Returns suggested rows, cols, and confidence
        """
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            
            # Analyze horizontal and vertical line densities
            h_projection = np.sum(edges, axis=1)
            v_projection = np.sum(edges, axis=0)
            
            # Find peaks in projections (indicate grid lines)
            h_peaks = GridLayoutAnalyzer._find_peaks(h_projection, min_distance=height//10)
            v_peaks = GridLayoutAnalyzer._find_peaks(v_projection, min_distance=width//10)
            
            # Estimate grid dimensions
            estimated_rows = len(h_peaks) - 1 if len(h_peaks) > 1 else 3
            estimated_cols = len(v_peaks) - 1 if len(v_peaks) > 1 else 3
            
            # Calculate confidence based on peak regularity
            confidence = GridLayoutAnalyzer._calculate_grid_confidence(h_peaks, v_peaks, height, width)
            
            return {
                'suggested_rows': max(1, min(6, estimated_rows)),  # Clamp between 1-6
                'suggested_cols': max(1, min(6, estimated_cols)),  # Clamp between 1-6
                'confidence': confidence,
                'h_peaks': len(h_peaks),
                'v_peaks': len(v_peaks)
            }
            
        except Exception as e:
            logger.error(f"Layout analysis failed: {str(e)}")
            return {
                'suggested_rows': 3,
                'suggested_cols': 3,
                'confidence': 0.0,
                'h_peaks': 0,
                'v_peaks': 0
            }
    
    @staticmethod
    def _find_peaks(projection: np.ndarray, min_distance: int) -> List[int]:
        """Find peaks in projection array"""
        peaks = []
        threshold = np.max(projection) * 0.3  # 30% of max value
        
        for i in range(1, len(projection) - 1):
            if (projection[i] > threshold and 
                projection[i] > projection[i-1] and 
                projection[i] > projection[i+1]):
                
                # Check minimum distance from previous peaks
                if not peaks or abs(i - peaks[-1]) >= min_distance:
                    peaks.append(i)
        
        return peaks
    
    @staticmethod
    def _calculate_grid_confidence(h_peaks: List[int], v_peaks: List[int], 
                                 height: int, width: int) -> float:
        """Calculate confidence in grid detection based on peak regularity"""
        if len(h_peaks) < 2 or len(v_peaks) < 2:
            return 0.0
        
        # Check spacing regularity
        h_spacings = [h_peaks[i+1] - h_peaks[i] for i in range(len(h_peaks)-1)]
        v_spacings = [v_peaks[i+1] - v_peaks[i] for i in range(len(v_peaks)-1)]
        
        # Calculate coefficient of variation (lower is more regular)
        h_cv = np.std(h_spacings) / np.mean(h_spacings) if h_spacings else 1.0
        v_cv = np.std(v_spacings) / np.mean(v_spacings) if v_spacings else 1.0
        
        # Convert to confidence (0-1, higher is better)
        h_confidence = max(0, 1 - h_cv)
        v_confidence = max(0, 1 - v_cv)
        
        return (h_confidence + v_confidence) / 2