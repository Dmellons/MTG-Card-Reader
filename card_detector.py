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
    """Detects and segments individual cards using shape recognition"""
    
    def __init__(self):
        self.expected_rows = settings.EXPECTED_GRID_ROWS
        self.expected_cols = settings.EXPECTED_GRID_COLS
        self.perspective_correction = settings.PERSPECTIVE_CORRECTION
        
        # Card detection parameters - made more flexible for scanned sheets
        self.min_card_area = 2000  # Minimum area for a card (reduced)
        self.max_card_area = 800000  # Maximum area for a card (increased)
        self.card_aspect_ratio_range = (0.55, 0.85)  # More flexible aspect ratio range
    
    def detect_cards_by_shape(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect individual cards by finding rectangular shapes that match card proportions
        Returns list of individual card images
        """
        try:
            logger.debug("Starting shape-based card detection")
            
            # Preprocess image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            logger.debug(f"Image size: {gray.shape}")
            
            # Apply bilateral filter to reduce noise while keeping edges sharp
            filtered = cv2.bilateralFilter(gray, 11, 17, 17)
            
            # Use multiple edge detection approaches and combine them
            edges = self._multi_scale_edge_detection(filtered)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            logger.debug(f"Found {len(contours)} total contours")
            
            # Filter and process contours to find card-like shapes
            card_contours = self._filter_card_contours(contours, image.shape)
            
            # Extract card regions from valid contours
            cards = self._extract_card_regions(image, card_contours)
            
            # Remove overlapping detections
            cards = self._remove_overlapping_cards(cards)
            
            logger.info(f"Shape-based detection found {len(cards)} cards")
            return cards
            
        except Exception as e:
            logger.error(f"Shape-based card detection failed: {str(e)}")
            return []
    
    def _multi_scale_edge_detection(self, gray_image: np.ndarray) -> np.ndarray:
        """Apply edge detection at multiple scales and combine results"""
        edges_combined = np.zeros_like(gray_image)
        
        # More sensitive Canny edge detection for scanned documents
        for low_thresh, high_thresh in [(20, 60), (40, 120), (60, 180)]:
            edges = cv2.Canny(gray_image, low_thresh, high_thresh, apertureSize=3)
            edges_combined = cv2.bitwise_or(edges_combined, edges)
        
        # Morphological operations to connect broken edges and fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_combined = cv2.morphologyEx(edges_combined, cv2.MORPH_CLOSE, kernel)
        
        # Additional dilation to strengthen edges
        edges_combined = cv2.dilate(edges_combined, kernel, iterations=1)
        
        return edges_combined
    
    def _filter_card_contours(self, contours: List, image_shape: Tuple) -> List:
        """Filter contours to find those that look like cards"""
        height, width = image_shape[:2]
        image_area = height * width
        
        card_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Check area constraints
            if not (self.min_card_area < area < min(self.max_card_area, image_area * 0.4)):
                continue
            
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Look for roughly rectangular shapes (4-6 vertices after approximation)
            if len(approx) < 4 or len(approx) > 8:
                continue
            
            # Check bounding rectangle properties
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Check aspect ratio (cards should be roughly rectangular)
            if not (self.card_aspect_ratio_range[0] <= aspect_ratio <= self.card_aspect_ratio_range[1]):
                continue
            
            # Check that contour fills reasonable amount of bounding rectangle
            rect_area = w * h
            fill_ratio = area / rect_area
            if fill_ratio < 0.5:  # More lenient fill ratio for scanned cards
                continue
            
            # Check for minimum size relative to image (even more lenient)
            if w < width * 0.03 or h < height * 0.03:
                continue
            
            # Additional check: ensure the shape is reasonably convex (more lenient)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity < 0.6:  # More lenient solidity for scanned cards
                continue
            
            card_contours.append({
                'contour': contour,
                'area': area,
                'bbox': (x, y, w, h),
                'aspect_ratio': aspect_ratio,
                'approx': approx
            })
        
        # Sort by area (largest first)
        card_contours.sort(key=lambda x: x['area'], reverse=True)
        
        logger.debug(f"Found {len(card_contours)} potential card contours")
        return card_contours
    
    def _extract_card_regions(self, image: np.ndarray, card_contours: List[Dict]) -> List[np.ndarray]:
        """Extract individual card regions from the image"""
        cards = []
        
        for card_info in card_contours:
            try:
                x, y, w, h = card_info['bbox']
                
                # Add small padding around the card
                padding = max(5, min(w, h) // 50)
                x_padded = max(0, x - padding)
                y_padded = max(0, y - padding)
                w_padded = min(image.shape[1] - x_padded, w + 2 * padding)
                h_padded = min(image.shape[0] - y_padded, h + 2 * padding)
                
                # Extract the card region
                card_region = image[y_padded:y_padded + h_padded, x_padded:x_padded + w_padded]
                
                if card_region.size > 0:
                    # Apply perspective correction if the card is skewed
                    corrected_card = self._correct_card_perspective(card_region, card_info['approx'])
                    cards.append(corrected_card)
                    
            except Exception as e:
                logger.warning(f"Failed to extract card region: {str(e)}")
                continue
        
        return cards
    
    def _correct_card_perspective(self, card_image: np.ndarray, approx_contour: np.ndarray) -> np.ndarray:
        """Apply perspective correction to straighten skewed cards"""
        try:
            if not self.perspective_correction or len(approx_contour) != 4:
                return card_image
            
            # Get the four corners of the approximated contour
            corners = approx_contour.reshape(4, 2)
            
            # Order corners: top-left, top-right, bottom-right, bottom-left
            corners = self._order_corners(corners)
            
            # Define target rectangle dimensions
            height, width = card_image.shape[:2]
            target_corners = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype=np.float32)
            
            # Calculate perspective transform matrix
            matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), target_corners)
            
            # Apply perspective correction
            corrected = cv2.warpPerspective(card_image, matrix, (width, height))
            
            return corrected
            
        except Exception as e:
            logger.debug(f"Perspective correction failed: {str(e)}")
            return card_image
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners as: top-left, top-right, bottom-right, bottom-left"""
        # Sum and difference of coordinates
        sum_coords = corners.sum(axis=1)
        diff_coords = np.diff(corners, axis=1)
        
        # Top-left has smallest sum, bottom-right has largest sum
        top_left = corners[np.argmin(sum_coords)]
        bottom_right = corners[np.argmax(sum_coords)]
        
        # Top-right has smallest difference, bottom-left has largest difference
        top_right = corners[np.argmin(diff_coords)]
        bottom_left = corners[np.argmax(diff_coords)]
        
        return np.array([top_left, top_right, bottom_right, bottom_left])
    
    def _remove_overlapping_cards(self, cards: List[np.ndarray]) -> List[np.ndarray]:
        """Remove cards that significantly overlap with others"""
        if len(cards) <= 1:
            return cards
        
        # For now, just return all cards - overlap detection would need bounding box tracking
        # This could be enhanced by tracking original positions and checking IoU
        return cards
    
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
            logger.debug(f"Attempting card detection (expected: {expected_rows}x{expected_cols})")
            
            # First try precise 3x3 detection for MTG scans
            if expected_rows == 3 and expected_cols == 3:
                precise_cards = self._detect_precise_3x3_grid(image)
                if len(precise_cards) == 9:
                    logger.info("Found perfect 3x3 grid with precise detection")
                    return precise_cards
            
            # Try border-based detection (most MTG-specific)
            border_cards = self.detect_cards_by_borders(image)
            if len(border_cards) > 0:
                logger.info(f"Border-based detection found {len(border_cards)} cards")
                return border_cards
            
            # Try shape-based detection as backup
            shape_cards = self.detect_cards_by_shape(image)
            if len(shape_cards) > 0:
                logger.info(f"Shape-based detection found {len(shape_cards)} cards")
                return shape_cards
            
            # Fallback to whitespace-based detection
            whitespace_cards = self._detect_cards_by_whitespace(image, expected_rows, expected_cols)
            if len(whitespace_cards) > 0:
                logger.info(f"Whitespace detection found {len(whitespace_cards)} cards")
                return whitespace_cards
            
            # Last resort: contour-based detection with grid constraints
            contour_cards = self._detect_cards_by_contours(image, expected_rows, expected_cols)
            if len(contour_cards) > 0:
                logger.info(f"Contour detection found {len(contour_cards)} cards")
                return contour_cards
            
            logger.warning("All detection methods failed")
            return []
            
        except Exception as e:
            logger.error(f"Card detection failed: {str(e)}")
            return []
    
    def _detect_precise_3x3_grid(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Precise detection for 3x3 MTG card grids with white separators
        """
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find horizontal and vertical separators
        horizontal_lines = self._find_grid_separators(gray, 'horizontal')
        vertical_lines = self._find_grid_separators(gray, 'vertical')
        
        logger.debug(f"Found {len(horizontal_lines)} horizontal and {len(vertical_lines)} vertical separators")
        
        # For 3x3 grid, we need 2 horizontal and 2 vertical separators
        if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
            # Sort lines
            horizontal_lines = sorted(horizontal_lines)
            vertical_lines = sorted(vertical_lines)
            
            # Take the 2 most prominent lines for clean 3x3 separation
            # Skip the first and last if we have more than 2
            if len(horizontal_lines) > 2:
                h_dividers = horizontal_lines[1:3]  # Skip edges
            else:
                h_dividers = horizontal_lines[:2]
                
            if len(vertical_lines) > 2:
                v_dividers = vertical_lines[1:3]  # Skip edges
            else:
                v_dividers = vertical_lines[:2]
            
            # Create 3x3 grid boundaries
            y_bounds = [0] + h_dividers + [height]
            x_bounds = [0] + v_dividers + [width]
            
            logger.debug(f"Grid bounds - Y: {y_bounds}, X: {x_bounds}")
            
            cards = []
            for row in range(3):
                for col in range(3):
                    y1, y2 = y_bounds[row], y_bounds[row + 1]
                    x1, x2 = x_bounds[col], x_bounds[col + 1]
                    
                    # Add padding to avoid white borders
                    padding_x = max(10, (x2 - x1) // 20)  # 5% padding
                    padding_y = max(10, (y2 - y1) // 20)  # 5% padding
                    
                    y1 += padding_y
                    y2 -= padding_y
                    x1 += padding_x
                    x2 -= padding_x
                    
                    if y2 > y1 and x2 > x1:
                        card = image[y1:y2, x1:x2]
                        if self._is_valid_card_region(card):
                            cards.append(card)
                            logger.debug(f"Extracted card {len(cards)}: {card.shape} at ({x1},{y1})")
                        
            return cards
        
        logger.debug("Not enough separators found for 3x3 grid")
        return []
    
    def _find_grid_separators(self, gray_image: np.ndarray, direction: str) -> List[int]:
        """
        Find white separator lines in the image by looking for thick white bands
        """
        height, width = gray_image.shape
        separators = []
        
        # For 3x3 grid, we expect exactly 2 separators in each direction
        # Focus on finding thick white bands (not thin lines or card borders)
        
        if direction == 'horizontal':
            # Expected positions for 3x3 grid
            expected_pos = [height // 3, 2 * height // 3]
            search_range = height // 6  # Search within 1/6 of image height
            
            for expected_y in expected_pos:
                best_y = None
                best_score = 0
                
                # Search around expected position
                start_y = max(0, expected_y - search_range)
                end_y = min(height, expected_y + search_range)
                
                for y in range(start_y, end_y, 5):  # Sample every 5 pixels
                    # Check multiple rows to ensure it's a thick white band
                    white_scores = []
                    for offset in range(-10, 11, 5):  # Check ±10 pixels around this position
                        check_y = y + offset
                        if 0 <= check_y < height:
                            row = gray_image[check_y, :]
                            white_ratio = np.sum(row > 230) / len(row)  # Very white pixels
                            white_scores.append(white_ratio)
                    
                    # Average white score across the band
                    avg_white_score = np.mean(white_scores) if white_scores else 0
                    
                    # Require consistently high whiteness
                    if avg_white_score > 0.8 and avg_white_score > best_score:
                        best_score = avg_white_score
                        best_y = y
                
                if best_y is not None:
                    separators.append(best_y)
                    
        elif direction == 'vertical':
            # Expected positions for 3x3 grid
            expected_pos = [width // 3, 2 * width // 3]
            search_range = width // 6  # Search within 1/6 of image width
            
            for expected_x in expected_pos:
                best_x = None
                best_score = 0
                
                # Search around expected position
                start_x = max(0, expected_x - search_range)
                end_x = min(width, expected_x + search_range)
                
                for x in range(start_x, end_x, 5):  # Sample every 5 pixels
                    # Check multiple columns to ensure it's a thick white band
                    white_scores = []
                    for offset in range(-10, 11, 5):  # Check ±10 pixels around this position
                        check_x = x + offset
                        if 0 <= check_x < width:
                            col = gray_image[:, check_x]
                            white_ratio = np.sum(col > 200) / len(col)  # More lenient for thin vertical gaps
                            white_scores.append(white_ratio)
                    
                    # Average white score across the band
                    avg_white_score = np.mean(white_scores) if white_scores else 0
                    
                    # Require consistently high whiteness (more lenient for vertical)
                    if avg_white_score > 0.6 and avg_white_score > best_score:
                        best_score = avg_white_score
                        best_x = x
                
                if best_x is not None:
                    separators.append(best_x)
        
        return sorted(separators)
    
    def detect_cards_by_borders(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect MTG cards by their distinctive border patterns
        MTG cards have outer border (white/black) + inner textured border
        """
        try:
            logger.debug("Starting border-based card detection")
            
            # Convert to different color spaces for border detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            logger.debug(f"Image size: {gray.shape}")
            
            # Detect card borders using multiple approaches
            border_mask = self._detect_card_borders(gray, hsv)
            
            # Find card contours from border mask
            contours, _ = cv2.findContours(border_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            logger.debug(f"Found {len(contours)} border contours")
            
            # Filter contours for card-like properties
            card_contours = self._filter_border_contours(contours, image.shape)
            
            # Extract card regions
            cards = self._extract_bordered_card_regions(image, card_contours)
            
            logger.info(f"Border-based detection found {len(cards)} cards")
            return cards
            
        except Exception as e:
            logger.error(f"Border-based card detection failed: {str(e)}")
            return []
    
    def _detect_card_borders(self, gray: np.ndarray, hsv: np.ndarray) -> np.ndarray:
        """Detect MTG card borders using color and texture analysis"""
        height, width = gray.shape
        
        # Method 1: Detect white borders
        white_mask = cv2.inRange(gray, 200, 255)
        
        # Method 2: Detect black borders  
        black_mask = cv2.inRange(gray, 0, 50)
        
        # Method 3: Detect edges that could be card boundaries
        # Use bilateral filter to preserve edges while reducing noise
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(filtered, 30, 80, apertureSize=3)
        
        # Method 4: Detect rectangular regions with high contrast
        # Look for areas where there's a significant color change (border transition)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2).astype(np.uint8)
        
        # Threshold gradient to find strong edges
        _, gradient_thresh = cv2.threshold(gradient_magnitude, 30, 255, cv2.THRESH_BINARY)
        
        # Combine all detection methods
        combined_mask = cv2.bitwise_or(white_mask, black_mask)
        combined_mask = cv2.bitwise_or(combined_mask, edges)
        combined_mask = cv2.bitwise_or(combined_mask, gradient_thresh)
        
        # Morphological operations to connect border segments
        # Use rectangular kernel to favor horizontal/vertical lines (card edges)
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        
        # Close gaps in horizontal and vertical directions
        h_closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_h)
        v_closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_v)
        closed = cv2.bitwise_or(h_closed, v_closed)
        
        # Fill small holes and connect nearby border segments
        kernel_fill = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        filled = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_fill)
        
        # Dilate to ensure we capture the full border area
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        final_mask = cv2.dilate(filled, kernel_dilate, iterations=2)
        
        return final_mask
    
    def _filter_border_contours(self, contours: List, image_shape: Tuple) -> List[Dict]:
        """Filter contours that look like MTG card borders"""
        height, width = image_shape[:2]
        image_area = height * width
        
        card_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # MTG cards should be a reasonable size relative to the image
            min_area = image_area * 0.01  # At least 1% of image
            max_area = image_area * 0.3   # At most 30% of image
            
            if not (min_area < area < max_area):
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio (MTG cards are ~63mm x 88mm = ~0.716 ratio)
            aspect_ratio = w / h
            if not (0.55 <= aspect_ratio <= 0.85):
                continue
            
            # Check minimum size
            if w < width * 0.05 or h < height * 0.05:
                continue
            
            # Check that the contour reasonably fills its bounding rectangle
            rect_area = w * h
            fill_ratio = area / rect_area
            if fill_ratio < 0.3:  # Very lenient for border detection
                continue
            
            # Approximate contour to polygon to check if it's roughly rectangular
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Should have 4-8 vertices for a rectangular-ish shape
            if len(approx) < 4 or len(approx) > 12:
                continue
            
            card_contours.append({
                'contour': contour,
                'area': area,
                'bbox': (x, y, w, h),
                'aspect_ratio': aspect_ratio,
                'approx': approx,
                'fill_ratio': fill_ratio
            })
        
        # Sort by area (largest first) and confidence metrics
        card_contours.sort(key=lambda x: (x['area'], x['fill_ratio']), reverse=True)
        
        logger.debug(f"Found {len(card_contours)} valid card border contours")
        return card_contours
    
    def _extract_bordered_card_regions(self, image: np.ndarray, card_contours: List[Dict]) -> List[np.ndarray]:
        """Extract card regions based on detected borders"""
        cards = []
        
        for card_info in card_contours:
            try:
                x, y, w, h = card_info['bbox']
                
                # Add padding around the detected border
                # This ensures we get the full card including any outer border
                padding_x = max(3, w // 100)  # Minimal padding proportional to card size
                padding_y = max(3, h // 100)
                
                x_padded = max(0, x - padding_x)
                y_padded = max(0, y - padding_y)
                w_padded = min(image.shape[1] - x_padded, w + 2 * padding_x)
                h_padded = min(image.shape[0] - y_padded, h + 2 * padding_y)
                
                # Extract the card region
                card_region = image[y_padded:y_padded + h_padded, x_padded:x_padded + w_padded]
                
                if card_region.size > 0:
                    # Apply perspective correction if needed
                    if len(card_info['approx']) == 4:
                        corrected_card = self._correct_card_perspective(card_region, card_info['approx'])
                        cards.append(corrected_card)
                    else:
                        cards.append(card_region)
                    
                    logger.debug(f"Extracted card: {w_padded}x{h_padded} at ({x_padded},{y_padded})")
                    
            except Exception as e:
                logger.warning(f"Failed to extract bordered card region: {str(e)}")
                continue
        
        return cards
    
    def detect_cards_smart(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Smart card detection that automatically chooses the best method
        """
        # Try border-based detection first (most MTG-specific)
        cards = self.detect_cards_by_borders(image)
        
        if len(cards) > 0:
            logger.info(f"Smart detection found {len(cards)} cards using border detection")
            return cards
        
        # Try shape-based detection as backup
        cards = self.detect_cards_by_shape(image)
        
        if len(cards) > 0:
            logger.info(f"Smart detection found {len(cards)} cards using shape detection")
            return cards
        
        # If both fail, fall back to grid detection
        return self.detect_card_grid_advanced(image)
    
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
    
    def _detect_cards_by_whitespace(self, image: np.ndarray, 
                                   expected_rows: int, expected_cols: int) -> List[np.ndarray]:
        """Detect cards by analyzing whitespace patterns between them"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Try multiple approaches for grid detection
            approaches = [
                {'threshold': 200, 'min_gap_h': height//40, 'min_gap_v': width//40},
                {'threshold': 180, 'min_gap_h': height//50, 'min_gap_v': width//50},
                {'threshold': 220, 'min_gap_h': height//30, 'min_gap_v': width//30},
            ]
            
            best_cards = []
            best_score = 0
            
            for approach in approaches:
                # Create binary image focusing on whitespace/background
                _, binary = cv2.threshold(gray, approach['threshold'], 255, cv2.THRESH_BINARY)
                
                # Get whitespace projections
                h_projection = np.sum(binary, axis=1) / width
                v_projection = np.sum(binary, axis=0) / height
                
                # Find whitespace gaps
                h_gaps = GridLayoutAnalyzer._find_whitespace_gaps(h_projection, min_gap_size=approach['min_gap_h'])
                v_gaps = GridLayoutAnalyzer._find_whitespace_gaps(v_projection, min_gap_size=approach['min_gap_v'])
                
                if not h_gaps and not v_gaps:
                    continue
                
                # Try the expected grid size first, then be flexible
                for priority, (target_rows, target_cols) in enumerate([(expected_rows, expected_cols), (3, 3), (2, 2), (4, 4)]):
                    # If we don't have enough gaps, try to force the grid
                    if len(h_gaps) + 1 != target_rows or len(v_gaps) + 1 != target_cols:
                        # Try uniform grid division as fallback
                        h_lines = [i * height // target_rows for i in range(target_rows + 1)]
                        v_lines = [i * width // target_cols for i in range(target_cols + 1)]
                    else:
                        # Use detected gaps
                        h_lines = [0] + [(gap[0] + gap[1]) // 2 for gap in h_gaps] + [height]
                        v_lines = [0] + [(gap[0] + gap[1]) // 2 for gap in v_gaps] + [width]
                    
                    h_lines.sort()
                    v_lines.sort()
                    
                    # Extract card regions
                    cards = []
                    for i in range(len(h_lines) - 1):
                        for j in range(len(v_lines) - 1):
                            y1 = h_lines[i]
                            y2 = h_lines[i + 1]
                            x1 = v_lines[j]
                            x2 = v_lines[j + 1]
                            
                            # Add small padding to avoid including borders
                            padding = 15
                            y1 = max(0, y1 + padding)
                            y2 = min(height, y2 - padding)
                            x1 = max(0, x1 + padding)
                            x2 = min(width, x2 - padding)
                            
                            if y2 > y1 and x2 > x1:
                                card_region = image[y1:y2, x1:x2]
                                if self._is_valid_card_region(card_region):
                                    cards.append(card_region)
                    
                    # Score this approach based on number of valid cards found
                    # Strongly prefer results that match expected grid size exactly
                    expected_total = target_rows * target_cols
                    if len(cards) == expected_total:
                        score = len(cards) + 20 - priority * 5  # Big bonus for exact match, prioritize expected grid
                    else:
                        score = len(cards) - abs(len(cards) - expected_total) * 2  # Higher penalty for mismatch
                    
                    if score > best_score:
                        best_score = score
                        best_cards = cards
                        logger.debug(f"Better result: {len(cards)} cards with {target_rows}x{target_cols} grid (score: {score:.1f})")
            
            logger.info(f"Whitespace detection found {len(best_cards)} valid card regions (best score: {best_score:.1f})")
            return best_cards
            
        except Exception as e:
            logger.error(f"Whitespace detection failed: {str(e)}")
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
        
        # Check minimum size - be more lenient for smaller scans
        min_size = 80  # Reduced from 100
        if height < min_size or width < min_size:
            logger.debug(f"Card region too small: {width}x{height} < {min_size}")
            return False
        
        # Check aspect ratio (MTG cards are roughly 2.5:3.5 ratio, but allow more variance)
        aspect_ratio = float(width) / height
        if not (0.4 < aspect_ratio < 2.5):  # More lenient range
            logger.debug(f"Card aspect ratio invalid: {aspect_ratio:.2f}")
            return False
        
        # Check that the region has sufficient content (not mostly white/black)
        gray = cv2.cvtColor(card_region, cv2.COLOR_BGR2GRAY) if len(card_region.shape) == 3 else card_region
        
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        total_pixels = height * width
        
        # Check if too much white (background) or black - be more lenient
        white_ratio = hist[240:].sum() / total_pixels  # Changed from 250 to 240
        black_ratio = hist[:15].sum() / total_pixels   # Changed from 5 to 15
        
        if white_ratio > 0.9 or black_ratio > 0.9:  # Changed from 0.8 to 0.9
            logger.debug(f"Card region mostly uniform color: white={white_ratio:.2f}, black={black_ratio:.2f}")
            return False
        
        # Check for sufficient detail/variance in the image
        variance = np.var(gray)
        if variance < 100:  # Very low variance indicates uniform region
            logger.debug(f"Card region has low variance: {variance:.1f}")
            return False
        
        logger.debug(f"Valid card region: {width}x{height}, ratio={aspect_ratio:.2f}, variance={variance:.1f}")
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
        Analyze page layout to determine likely grid structure using whitespace detection
        Returns suggested rows, cols, and confidence
        """
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Focus on detecting whitespace (background) between cards
            # Create binary image where white/light areas are foreground
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            
            # Analyze whitespace patterns
            h_projection = np.sum(binary, axis=1) / width  # Average whiteness per row
            v_projection = np.sum(binary, axis=0) / height  # Average whiteness per column
            
            # Find gaps (high whitespace areas) that indicate card boundaries
            h_gaps = GridLayoutAnalyzer._find_whitespace_gaps(h_projection, min_gap_size=height//40)
            v_gaps = GridLayoutAnalyzer._find_whitespace_gaps(v_projection, min_gap_size=width//40)
            
            # Estimate grid dimensions based on gaps between cards
            estimated_rows = len(h_gaps) + 1 if h_gaps else 3
            estimated_cols = len(v_gaps) + 1 if v_gaps else 3
            
            # For typical MTG card sheets, clamp to reasonable values
            estimated_rows = max(1, min(5, estimated_rows))
            estimated_cols = max(1, min(5, estimated_cols))
            
            # Calculate confidence based on gap regularity and strength
            confidence = GridLayoutAnalyzer._calculate_whitespace_confidence(h_gaps, v_gaps, h_projection, v_projection)
            
            logger.debug(f"Layout analysis: {estimated_rows}x{estimated_cols}, conf={confidence:.2f}, h_gaps={len(h_gaps)}, v_gaps={len(v_gaps)}")
            
            return {
                'suggested_rows': estimated_rows,
                'suggested_cols': estimated_cols,
                'confidence': confidence,
                'h_peaks': len(h_gaps),
                'v_peaks': len(v_gaps)
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
    def _find_whitespace_gaps(projection: np.ndarray, min_gap_size: int) -> List[Tuple[int, int]]:
        """Find continuous whitespace gaps that indicate boundaries between cards"""
        gaps = []
        
        # Use adaptive threshold based on projection statistics
        # Lower threshold to detect smaller gaps
        threshold = np.mean(projection) + np.std(projection) * 0.2
        
        in_gap = False
        gap_start = 0
        
        for i, value in enumerate(projection):
            if value > threshold and not in_gap:
                # Start of a gap
                in_gap = True
                gap_start = i
            elif value <= threshold and in_gap:
                # End of a gap
                gap_size = i - gap_start
                if gap_size >= min_gap_size:
                    gap_center = gap_start + gap_size // 2
                    gaps.append((gap_start, i - 1))
                in_gap = False
        
        # Handle case where gap extends to end
        if in_gap:
            gap_size = len(projection) - gap_start
            if gap_size >= min_gap_size:
                gaps.append((gap_start, len(projection) - 1))
        
        return gaps
    
    @staticmethod
    def _calculate_whitespace_confidence(h_gaps: List[Tuple[int, int]], v_gaps: List[Tuple[int, int]], 
                                       h_projection: np.ndarray, v_projection: np.ndarray) -> float:
        """Calculate confidence based on whitespace gap quality"""
        if not h_gaps or not v_gaps:
            return 0.1
        
        # Check gap regularity (spacing between gaps)
        h_centers = [(gap[0] + gap[1]) // 2 for gap in h_gaps]
        v_centers = [(gap[0] + gap[1]) // 2 for gap in v_gaps]
        
        confidence = 0.5  # Base confidence
        
        # Bonus for reasonable number of gaps (2-4 for typical grids)
        if 1 <= len(h_gaps) <= 4 and 1 <= len(v_gaps) <= 4:
            confidence += 0.2
        
        # Bonus for regular spacing
        if len(h_centers) >= 2:
            h_spacings = [h_centers[i+1] - h_centers[i] for i in range(len(h_centers)-1)]
            h_regularity = 1 - (np.std(h_spacings) / np.mean(h_spacings)) if np.mean(h_spacings) > 0 else 0
            confidence += h_regularity * 0.15
        
        if len(v_centers) >= 2:
            v_spacings = [v_centers[i+1] - v_centers[i] for i in range(len(v_centers)-1)]
            v_regularity = 1 - (np.std(v_spacings) / np.mean(v_spacings)) if np.mean(v_spacings) > 0 else 0
            confidence += v_regularity * 0.15
        
        return min(1.0, confidence)
    
    @staticmethod
    def _find_peaks(projection: np.ndarray, min_distance: int) -> List[int]:
        """Find peaks in projection array (legacy method)"""
        peaks = []
        threshold = np.max(projection) * 0.6  # Increased threshold to 60%
        
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