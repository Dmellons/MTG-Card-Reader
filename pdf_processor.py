"""
PDF Processing Module for MTG Card Processing System
Handles both scanned and text-based PDFs
"""
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from typing import List, Tuple, Dict
import logging
from config import settings

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Main PDF processing class"""
    
    def __init__(self):
        self.dpi_scanned = settings.PDF_DPI_SCANNED
        self.dpi_text = settings.PDF_DPI_TEXT
    
    def detect_pdf_type(self, pdf_path: str) -> str:
        """
        Determine if PDF contains selectable text or is image-based
        Returns: 'text_based' or 'scanned'
        """
        try:
            doc = fitz.open(pdf_path)
            total_text_area = 0
            total_page_area = 0
            
            # Sample first 3 pages for efficiency
            sample_pages = min(3, len(doc))
            
            for page_num in range(sample_pages):
                page = doc.load_page(page_num)
                page_rect = page.rect
                total_page_area += page_rect.width * page_rect.height
                
                # Get text blocks with position information
                text_blocks = page.get_text("dict")
                for block in text_blocks["blocks"]:
                    if "lines" in block:
                        bbox = block["bbox"]
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        total_text_area += area
            
            doc.close()
            
            # Calculate text coverage ratio
            text_ratio = total_text_area / total_page_area if total_page_area > 0 else 0
            pdf_type = "text_based" if text_ratio > 0.1 else "scanned"
            
            logger.info(f"PDF type detected: {pdf_type} (text ratio: {text_ratio:.3f})")
            return pdf_type
            
        except Exception as e:
            logger.error(f"Error detecting PDF type: {str(e)}")
            return "scanned"  # Default to scanned if detection fails
    
    def convert_pdf_to_images(self, pdf_path: str, pdf_type: str = None) -> List[np.ndarray]:
        """
        Convert PDF pages to high-quality images
        Returns list of OpenCV images (BGR format)
        """
        if pdf_type is None:
            pdf_type = self.detect_pdf_type(pdf_path)
        
        # Choose DPI based on PDF type
        dpi = self.dpi_text if pdf_type == "text_based" else self.dpi_scanned
        
        try:
            doc = fitz.open(pdf_path)
            images = []
            
            logger.info(f"Converting {len(doc)} pages at {dpi} DPI")
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Create transformation matrix for high-quality rendering
                mat = fitz.Matrix(dpi/72, dpi/72)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                # Convert to PNG bytes then to OpenCV format
                img_data = pix.tobytes("png")
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    images.append(img)
                    logger.debug(f"Converted page {page_num + 1}, shape: {img.shape}")
                else:
                    logger.warning(f"Failed to convert page {page_num + 1}")
            
            doc.close()
            logger.info(f"Successfully converted {len(images)} pages")
            return images
            
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            return []
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract text with position information from text-based PDFs
        Returns list of text blocks with coordinates
        """
        try:
            doc = fitz.open(pdf_path)
            text_data = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_dict = page.get_text("dict")
                
                page_text = {
                    'page_number': page_num,
                    'width': page.rect.width,
                    'height': page.rect.height,
                    'blocks': []
                }
                
                for block in page_dict["blocks"]:
                    if "lines" in block:  # Text block
                        block_text = ""
                        for line in block["lines"]:
                            for span in line["spans"]:
                                block_text += span["text"] + " "
                        
                        if block_text.strip():
                            page_text['blocks'].append({
                                'text': block_text.strip(),
                                'bbox': block["bbox"],
                                'confidence': 1.0  # Text extraction has high confidence
                            })
                
                text_data.append(page_text)
            
            doc.close()
            return text_data
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return []

class ImagePreprocessor:
    """Image preprocessing utilities for better OCR and card detection"""
    
    @staticmethod
    def enhance_image_quality(image: np.ndarray) -> np.ndarray:
        """Enhance image quality for better OCR results"""
        try:
            # Convert to PIL for enhancement operations
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(pil_image)
            enhanced = enhancer.enhance(1.2)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(enhanced)
            sharpened = enhancer.enhance(1.1)
            
            # Convert back to OpenCV format
            result = cv2.cvtColor(np.array(sharpened), cv2.COLOR_RGB2BGR)
            return result
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {str(e)}")
            return image
    
    @staticmethod
    def auto_perspective_correction(image: np.ndarray) -> np.ndarray:
        """
        Detect and correct perspective distortion automatically
        Useful for scanned documents that aren't perfectly aligned
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort contours by area, largest first
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Look for a quadrilateral (4-sided shape)
            for contour in contours[:10]:  # Check top 10 largest contours
                # Approximate contour
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # If we found a quadrilateral with significant area
                if len(approx) == 4 and cv2.contourArea(approx) > image.shape[0] * image.shape[1] * 0.1:
                    logger.debug("Found quadrilateral for perspective correction")
                    return ImagePreprocessor.four_point_transform(image, approx.reshape(4, 2))
            
            logger.debug("No suitable quadrilateral found, returning original image")
            return image
            
        except Exception as e:
            logger.warning(f"Perspective correction failed: {str(e)}")
            return image
    
    @staticmethod
    def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Apply perspective transform using four corner points"""
        try:
            # Order points: top-left, top-right, bottom-right, bottom-left
            rect = ImagePreprocessor.order_points(pts)
            (tl, tr, br, bl) = rect
            
            # Compute width and height of new image
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            
            # Destination points for the transform
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype="float32")
            
            # Compute perspective transform matrix and apply it
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
            
            return warped
            
        except Exception as e:
            logger.warning(f"Four point transform failed: {str(e)}")
            return image
    
    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        """Order points in clockwise order starting with top-left"""
        rect = np.zeros((4, 2), dtype="float32")
        
        # Sum and difference to find corners
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        # Top-left will have smallest sum, bottom-right will have largest sum
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Top-right will have smallest difference, bottom-left will have largest difference
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    @staticmethod
    def adaptive_threshold(image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding for better text extraction"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Apply adaptive threshold
            thresh = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            return thresh
            
        except Exception as e:
            logger.warning(f"Adaptive threshold failed: {str(e)}")
            return image

def validate_image(image: np.ndarray) -> bool:
    """Validate that an image is suitable for processing"""
    if image is None:
        return False
    
    # Check dimensions
    if len(image.shape) not in [2, 3]:
        return False
    
    height, width = image.shape[:2]
    
    # Check minimum size (cards should be reasonably large)
    if height < 100 or width < 100:
        return False
    
    # Check maximum size (avoid processing extremely large images)
    if height > 10000 or width > 10000:
        return False
    
    return True