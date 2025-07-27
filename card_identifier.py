"""
Card Identification Module
Combines API lookup, image matching, and fuzzy matching for robust card identification
"""
import requests
import time
import cv2
import numpy as np
import imagehash
from PIL import Image
import json
import logging
import re
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher
from config import settings

logger = logging.getLogger(__name__)

class MTGCardIdentifier:
    """Main card identification class using Scryfall API"""
    
    def __init__(self):
        self.api_base = settings.SCRYFALL_API_BASE
        self.rate_limit_delay = settings.API_RATE_LIMIT_DELAY
        self.timeout = settings.API_TIMEOUT
        self.last_request_time = 0
        
        # Setup session with proper headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MTGCardProcessor/1.0 (Educational Project)',
            'Accept': 'application/json'
        })
        
        # Cache for recent API calls
        self.api_cache = {}
        self.cache_max_size = 1000
    
    def _rate_limit(self):
        """Enforce Scryfall API rate limiting (10 requests per second max)"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def identify_by_name(self, card_name: str, set_code: str = None) -> Optional[Dict]:
        """
        Identify card using fuzzy name matching via Scryfall API
        Returns complete card data if found
        """
        if not card_name or len(card_name.strip()) < 2:
            return None
        
        cache_key = f"{card_name.lower()}:{set_code or 'any'}"
        if cache_key in self.api_cache:
            logger.debug(f"Cache hit for {card_name}")
            return self.api_cache[cache_key]
        
        try:
            self._rate_limit()
            
            params = {'fuzzy': card_name.strip()}
            if set_code:
                params['set'] = set_code.lower()
            
            logger.debug(f"API request for card: {card_name}")
            response = self.session.get(
                f"{self.api_base}/cards/named", 
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                card_data = response.json()
                self._cache_result(cache_key, card_data)
                logger.info(f"Successfully identified: {card_data.get('name', 'Unknown')}")
                return card_data
            
            elif response.status_code == 404:
                logger.warning(f"Card not found: {card_name}")
                self._cache_result(cache_key, None)
                return None
            
            elif response.status_code == 429:
                # Rate limit exceeded
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limit hit, waiting {retry_after} seconds")
                time.sleep(retry_after)
                return self.identify_by_name(card_name, set_code)
            
            else:
                logger.error(f"API error {response.status_code}: {response.text}")
                return None
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during API request: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during card identification: {str(e)}")
            return None
    
    def search_cards_advanced(self, query: str, include_extras: bool = False) -> Optional[List[Dict]]:
        """
        Advanced card search with complex queries
        Example query: "type:creature color:red cmc:3"
        """
        try:
            self._rate_limit()
            
            params = {
                'q': query,
                'unique': 'cards' if not include_extras else 'prints',
                'order': 'name'
            }
            
            response = self.session.get(
                f"{self.api_base}/cards/search",
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('data', [])
            else:
                logger.warning(f"Search failed: {response.status_code}")
                return None
        
        except Exception as e:
            logger.error(f"Advanced search failed: {str(e)}")
            return None
    
    def get_card_by_scryfall_id(self, scryfall_id: str) -> Optional[Dict]:
        """Get card data by Scryfall ID"""
        try:
            self._rate_limit()
            
            response = self.session.get(
                f"{self.api_base}/cards/{scryfall_id}",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Card ID not found: {scryfall_id}")
                return None
        
        except Exception as e:
            logger.error(f"Error fetching card by ID: {str(e)}")
            return None
    
    def get_random_card(self, query: str = None) -> Optional[Dict]:
        """Get a random card, optionally matching a query"""
        try:
            self._rate_limit()
            
            params = {}
            if query:
                params['q'] = query
            
            response = self.session.get(
                f"{self.api_base}/cards/random",
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            
        except Exception as e:
            logger.error(f"Random card request failed: {str(e)}")
        
        return None
    
    def _cache_result(self, key: str, result: Optional[Dict]):
        """Cache API results with size management"""
        if len(self.api_cache) >= self.cache_max_size:
            # Remove oldest entries (simple FIFO)
            oldest_keys = list(self.api_cache.keys())[:100]
            for old_key in oldest_keys:
                del self.api_cache[old_key]
        
        self.api_cache[key] = result

class ImageMatcher:
    """Image-based card matching using perceptual hashing and feature detection"""
    
    def __init__(self, card_database_path: str = None):
        self.hash_threshold = settings.PERCEPTUAL_HASH_THRESHOLD
        self.min_feature_matches = settings.FEATURE_MATCH_MIN_MATCHES
        
        # Initialize ORB detector for feature matching
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Load precomputed hash database if available
        self.hash_database = self._load_hash_database(card_database_path)
        
        # SIFT detector for more robust feature matching (if available)
        try:
            self.sift = cv2.SIFT_create()
            self.sift_available = True
        except AttributeError:
            self.sift_available = False
            logger.warning("SIFT detector not available, using ORB only")
    
    def _load_hash_database(self, db_path: str) -> Dict[str, str]:
        """Load precomputed perceptual hash database"""
        if not db_path:
            return {}
        
        try:
            with open(db_path, 'r') as f:
                hash_db = json.load(f)
            logger.info(f"Loaded {len(hash_db)} card hashes from database")
            return hash_db
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load hash database: {str(e)}")
            return {}
    
    def match_by_perceptual_hash(self, query_image: np.ndarray, 
                                threshold: int = None) -> List[Tuple[str, float]]:
        """
        Match cards using perceptual hashing
        Returns list of (card_id, confidence) tuples
        """
        if threshold is None:
            threshold = self.hash_threshold
        
        try:
            # Convert to PIL Image for hashing
            if len(query_image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(query_image)
            
            # Calculate perceptual hash
            query_hash = imagehash.phash(pil_image, hash_size=8)
            
            matches = []
            for card_id, stored_hash_str in self.hash_database.items():
                try:
                    stored_hash = imagehash.hex_to_hash(stored_hash_str)
                    distance = query_hash - stored_hash
                    
                    if distance <= threshold:
                        # Convert distance to confidence (0-1 scale)
                        confidence = 1.0 - (distance / 64.0)
                        matches.append((card_id, confidence))
                
                except ValueError:
                    continue  # Skip invalid hash entries
            
            # Sort by confidence
            matches.sort(key=lambda x: x[1], reverse=True)
            logger.debug(f"Perceptual hash found {len(matches)} matches")
            return matches
        
        except Exception as e:
            logger.error(f"Perceptual hash matching failed: {str(e)}")
            return []
    
    def match_by_features(self, query_image: np.ndarray, 
                         reference_images: Dict[str, np.ndarray]) -> List[Tuple[str, float]]:
        """
        Match using ORB/SIFT feature detection
        Returns list of (card_id, confidence) tuples
        """
        try:
            # Convert to grayscale if needed
            if len(query_image.shape) == 3:
                query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
            else:
                query_gray = query_image
            
            # Detect keypoints and descriptors
            if self.sift_available:
                query_kp, query_desc = self.sift.detectAndCompute(query_gray, None)
            else:
                query_kp, query_desc = self.orb.detectAndCompute(query_gray, None)
            
            if query_desc is None or len(query_desc) < 10:
                logger.warning("Insufficient features detected in query image")
                return []
            
            matches = []
            for ref_id, ref_image in reference_images.items():
                try:
                    # Convert reference image
                    if len(ref_image.shape) == 3:
                        ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
                    else:
                        ref_gray = ref_image
                    
                    # Detect features in reference image
                    if self.sift_available:
                        ref_kp, ref_desc = self.sift.detectAndCompute(ref_gray, None)
                    else:
                        ref_kp, ref_desc = self.orb.detectAndCompute(ref_gray, None)
                    
                    if ref_desc is None or len(ref_desc) < 10:
                        continue
                    
                    # Match features
                    if self.sift_available:
                        # Use FLANN matcher for SIFT
                        FLANN_INDEX_KDTREE = 1
                        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                        search_params = dict(checks=50)
                        flann = cv2.FlannBasedMatcher(index_params, search_params)
                        
                        matches_raw = flann.knnMatch(query_desc, ref_desc, k=2)
                        
                        # Apply Lowe's ratio test
                        good_matches = []
                        for match_pair in matches_raw:
                            if len(match_pair) == 2:
                                m, n = match_pair
                                if m.distance < 0.7 * n.distance:
                                    good_matches.append(m)
                        
                        feature_matches = good_matches
                    else:
                        # Use BFMatcher for ORB
                        feature_matches = self.bf_matcher.match(query_desc, ref_desc)
                        feature_matches = sorted(feature_matches, key=lambda x: x.distance)
                    
                    if len(feature_matches) >= self.min_feature_matches:
                        # Calculate confidence based on number and quality of matches
                        if self.sift_available:
                            avg_distance = sum(m.distance for m in feature_matches) / len(feature_matches)
                            confidence = min(1.0, len(feature_matches) / 50.0) * (1.0 - min(1.0, avg_distance / 100.0))
                        else:
                            confidence = min(1.0, len(feature_matches) / max(len(query_desc), len(ref_desc)))
                        
                        matches.append((ref_id, confidence))
                
                except Exception as e:
                    logger.warning(f"Feature matching failed for {ref_id}: {str(e)}")
                    continue
            
            matches.sort(key=lambda x: x[1], reverse=True)
            logger.debug(f"Feature matching found {len(matches)} matches")
            return matches
        
        except Exception as e:
            logger.error(f"Feature matching failed: {str(e)}")
            return []
    
    def compute_image_hash(self, image: np.ndarray) -> str:
        """Compute perceptual hash for an image"""
        try:
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            phash = imagehash.phash(pil_image, hash_size=8)
            return str(phash)
        
        except Exception as e:
            logger.error(f"Hash computation failed: {str(e)}")
            return ""

class FuzzyMatcher:
    """Fuzzy string matching for card names and text"""
    
    @staticmethod
    def fuzzy_match_name(query_name: str, candidate_names: List[str], 
                        threshold: float = 0.7) -> List[Tuple[str, float]]:
        """
        Fuzzy match card names using sequence matching
        Returns list of (name, similarity) tuples
        """
        if not query_name or not candidate_names:
            return []
        
        query_clean = FuzzyMatcher._normalize_name(query_name)
        matches = []
        
        for candidate in candidate_names:
            candidate_clean = FuzzyMatcher._normalize_name(candidate)
            
            # Use SequenceMatcher for similarity
            similarity = SequenceMatcher(None, query_clean, candidate_clean).ratio()
            
            if similarity >= threshold:
                matches.append((candidate, similarity))
        
        # Sort by similarity
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize card name for better matching"""
        if not name:
            return ""
        
        # Convert to lowercase
        normalized = name.lower().strip()
        
        # Remove punctuation and extra spaces
        import string
        normalized = ''.join(char for char in normalized if char not in string.punctuation)
        normalized = ' '.join(normalized.split())
        
        # Handle common OCR errors
        ocr_corrections = {
            'ae': 'æ',
            'oe': 'œ',
            '0': 'o',
            '1': 'l',
            '5': 's',
            '8': 'b'
        }
        
        for error, correction in ocr_corrections.items():
            normalized = normalized.replace(error, correction)
        
        return normalized

class HybridIdentifier:
    """Combines multiple identification methods for robust card identification"""
    
    def __init__(self):
        self.api_identifier = MTGCardIdentifier()
        self.image_matcher = ImageMatcher()
        self.fuzzy_matcher = FuzzyMatcher()
        
        # Confidence weights for different methods
        self.method_weights = {
            'api_exact': 1.0,
            'api_fuzzy': 0.9,
            'image_hash': 0.7,
            'image_features': 0.8,
            'fuzzy_name': 0.6
        }
    
    def identify_card_comprehensive(self, card_image: np.ndarray, 
                                  extracted_name: str = None,
                                  extracted_text: str = None,
                                  set_hint: str = None) -> Dict:
        """
        Comprehensive card identification using all available methods
        Returns best match with confidence score and method used
        """
        identification_results = []
        
        # Method 1: API identification with extracted name
        if extracted_name and len(extracted_name.strip()) > 2:
            api_result = self._try_api_identification(extracted_name, set_hint)
            if api_result:
                identification_results.append(api_result)
        
        # Method 2: Perceptual hash matching
        hash_results = self._try_hash_matching(card_image)
        identification_results.extend(hash_results)
        
        # Method 3: Feature matching (if reference images available)
        # This would require a database of reference images
        # feature_results = self._try_feature_matching(card_image)
        # identification_results.extend(feature_results)
        
        # Method 4: Text-based fuzzy matching
        if extracted_text:
            text_results = self._try_text_matching(extracted_text)
            identification_results.extend(text_results)
        
        # Select best result
        if identification_results:
            best_result = max(identification_results, key=lambda x: x['confidence'])
            
            # Add metadata about the identification process
            best_result['all_methods_tried'] = len(identification_results)
            best_result['alternative_matches'] = [
                r for r in identification_results if r != best_result
            ][:3]  # Top 3 alternatives
            
            return best_result
        
        return {
            'success': False,
            'confidence': 0.0,
            'method': 'none',
            'card_data': None,
            'message': 'No identification method succeeded'
        }
    
    def _try_api_identification(self, card_name: str, set_hint: str = None) -> Optional[Dict]:
        """Try identification via Scryfall API"""
        try:
            # First try with set hint if available
            if set_hint:
                card_data = self.api_identifier.identify_by_name(card_name, set_hint)
                if card_data:
                    return {
                        'success': True,
                        'confidence': self.method_weights['api_exact'],
                        'method': 'api_exact_set',
                        'card_data': card_data
                    }
            
            # Try without set constraint
            card_data = self.api_identifier.identify_by_name(card_name)
            if card_data:
                confidence = self.method_weights['api_exact']
                method = 'api_exact'
                
                # Lower confidence if name doesn't match exactly
                if card_name and card_data.get('name', '').lower() != card_name.lower():
                    confidence = self.method_weights['api_fuzzy']
                    method = 'api_fuzzy'
                
                return {
                    'success': True,
                    'confidence': confidence,
                    'method': method,
                    'card_data': card_data
                }
        
        except Exception as e:
            logger.warning(f"API identification failed: {str(e)}")
        
        return None
    
    def _try_hash_matching(self, card_image: np.ndarray) -> List[Dict]:
        """Try identification via perceptual hash matching"""
        results = []
        
        try:
            hash_matches = self.image_matcher.match_by_perceptual_hash(card_image)
            
            for card_id, similarity in hash_matches[:3]:  # Top 3 matches
                # Would need to convert card_id to full card data
                # This requires a mapping from hash DB card IDs to Scryfall data
                results.append({
                    'success': True,
                    'confidence': similarity * self.method_weights['image_hash'],
                    'method': 'image_hash',
                    'card_data': {'id': card_id, 'name': f'Hash_Match_{card_id}'},
                    'hash_similarity': similarity
                })
        
        except Exception as e:
            logger.warning(f"Hash matching failed: {str(e)}")
        
        return results
    
    def _try_text_matching(self, extracted_text: str) -> List[Dict]:
        """Try identification using extracted text content"""
        results = []
        
        try:
            # Extract potential card names from text
            potential_names = self._extract_potential_names(extracted_text)
            
            for name in potential_names:
                if name and len(name) > 3:  # Reasonable minimum length
                    api_result = self._try_api_identification(name)
                    if api_result:
                        # Lower confidence since this is indirect
                        api_result['confidence'] *= self.method_weights['fuzzy_name']
                        api_result['method'] = 'text_derived'
                        results.append(api_result)
        
        except Exception as e:
            logger.warning(f"Text-based matching failed: {str(e)}")
        
        return results
    
    def _extract_potential_names(self, text: str) -> List[str]:
        """Extract potential card names from OCR text"""
        if not text:
            return []
        
        # Split text into lines and words
        lines = text.split('\n')
        potential_names = []
        
        for line in lines:
            line = line.strip()
            
            # Skip lines that are clearly not names
            if (len(line) < 3 or 
                line.isdigit() or 
                re.match(r'^[\{\}/\*\d\s]+$', line)):
                continue
            
            # Clean the line
            cleaned = re.sub(r'[^\w\s\-\']', '', line)
            cleaned = ' '.join(cleaned.split())
            
            if cleaned and len(cleaned) > 2:
                potential_names.append(cleaned)
        
        return potential_names[:5]  # Limit to top 5 candidates

class CardDatabaseBuilder:
    """Builds and maintains local card databases for faster matching"""
    
    def __init__(self, api_identifier: MTGCardIdentifier):
        self.api = api_identifier
        self.hash_db_path = "./data/card_hashes.json"
        self.name_db_path = "./data/card_names.json"
    
    def build_hash_database(self, card_images_dir: str):
        """Build perceptual hash database from card images"""
        import os
        
        hash_database = {}
        image_matcher = ImageMatcher()
        
        try:
            for filename in os.listdir(card_images_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(card_images_dir, filename)
                    
                    try:
                        image = cv2.imread(image_path)
                        if image is not None:
                            card_hash = image_matcher.compute_image_hash(image)
                            card_id = os.path.splitext(filename)[0]
                            hash_database[card_id] = card_hash
                            
                    except Exception as e:
                        logger.warning(f"Failed to hash {filename}: {str(e)}")
            
            # Save database
            with open(self.hash_db_path, 'w') as f:
                json.dump(hash_database, f)
            
            logger.info(f"Built hash database with {len(hash_database)} entries")
            return hash_database
        
        except Exception as e:
            logger.error(f"Failed to build hash database: {str(e)}")
            return {}
    
    def build_name_database(self, max_cards: int = 10000):
        """Build searchable name database from Scryfall"""
        try:
            # Get all card names from Scryfall
            # This is a simplified version - in practice you'd use bulk data
            response = self.api.session.get(f"{self.api.api_base}/catalog/card-names")
            
            if response.status_code == 200:
                data = response.json()
                card_names = data.get('data', [])
                
                # Save to file
                with open(self.name_db_path, 'w') as f:
                    json.dump(card_names, f)
                
                logger.info(f"Built name database with {len(card_names)} entries")
                return card_names
        
        except Exception as e:
            logger.error(f"Failed to build name database: {str(e)}")
        
        return []
    
    def update_databases(self):
        """Update local databases with latest data"""
        logger.info("Updating card databases...")
        
        # Update name database
        self.build_name_database()
        
        # Hash database updates would happen when new images are processed
        logger.info("Database update complete")

# Utility functions
def validate_identification_result(result: Dict) -> bool:
    """Validate that an identification result is reasonable"""
    if not result or not result.get('success'):
        return False
    
    confidence = result.get('confidence', 0)
    if confidence < 0.3:  # Minimum confidence threshold
        return False
    
    card_data = result.get('card_data')
    if not card_data:
        return False
    
    # Check that card data has reasonable fields
    name = card_data.get('name', '')
    if not name or len(name) < 2:
        return False
    
    return True

def merge_identification_results(results: List[Dict]) -> Dict:
    """Merge multiple identification results into a consensus"""
    if not results:
        return {'success': False, 'confidence': 0.0}
    
    # Group results by card name
    name_groups = {}
    for result in results:
        if not validate_identification_result(result):
            continue
        
        name = result['card_data'].get('name', '').lower()
        if name not in name_groups:
            name_groups[name] = []
        name_groups[name].append(result)
    
    if not name_groups:
        return {'success': False, 'confidence': 0.0}
    
    # Find consensus
    best_group = max(name_groups.values(), key=len)
    
    if len(best_group) == 1:
        return best_group[0]
    
    # Multiple results for same card - combine confidences
    total_confidence = sum(r['confidence'] for r in best_group)
    avg_confidence = total_confidence / len(best_group)
    
    best_result = max(best_group, key=lambda x: x['confidence'])
    best_result['confidence'] = min(1.0, avg_confidence * 1.2)  # Boost for consensus
    best_result['method'] = 'consensus'
    best_result['supporting_methods'] = [r['method'] for r in best_group]
    
    return best_result

class CardDatabaseBuilder:
    """Builds and maintains local card databases for faster matching"""
    
    def __init__(self, api_identifier: MTGCardIdentifier):
        self.api = api_identifier
        self.hash_db_path = "./data/card_hashes.json"
        self.name_db_path = "./data/card_names.json"
    
    def build_hash_database(self, card_images_dir: str):
        """Build perceptual hash database from card images"""
        import os
        
        hash_database = {}
        image_matcher = ImageMatcher()
        
        try:
            for filename in os.listdir(card_images_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(card_images_dir, filename)
                    
                    try:
                        image = cv2.imread(image_path)
                        if image is not None:
                            card_hash = image_matcher.compute_image_hash(image)
                            card_id = os.path.splitext(filename)[0]
                            hash_database[card_id] = card_hash
                            
                    except Exception as e:
                        logger.warning(f"Failed to hash {filename}: {str(e)}")
            
            # Save database
            with open(self.hash_db_path, 'w') as f:
                json.dump(hash_database, f)
            
            logger.info(f"Built hash database with {len(hash_database)} entries")
            return hash_database
        
        except Exception as e:
            logger.error(f"Failed to build hash database: {str(e)}")
            return {}
    
    def build_name_database(self, max_cards: int = 10000):
        """Build searchable name database from Scryfall"""
        try:
            # Get all card names from Scryfall
            # This is a simplified version - in practice you'd use bulk data
            response = self.api.session.get(f"{self.api.api_base}/catalog/card-names")
            
            if response.status_code == 200:
                data = response.json()
                card_names = data.get('data', [])
                
                # Save to file
                with open(self.name_db_path, 'w') as f:
                    json.dump(card_names, f)
                
                logger.info(f"Built name database with {len(card_names)} entries")
                return card_names
        
        except Exception as e:
            logger.error(f"Failed to build name database: {str(e)}")
        
        return []
    
    def update_databases(self):
        """Update local databases with latest data"""
        logger.info("Updating card databases...")
        
        # Update name database
        self.build_name_database()
        
        # Hash database updates would happen when new images are processed
        logger.info("Database update complete")

# Utility functions
def validate_identification_result(result: Dict) -> bool:
    """Validate that an identification result is reasonable"""
    if not result or not result.get('success'):
        return False
    
    confidence = result.get('confidence', 0)
    if confidence < 0.3:  # Minimum confidence threshold
        return False
    
    card_data = result.get('card_data')
    if not card_data:
        return False
    
    # Check that card data has reasonable fields
    name = card_data.get('name', '')
    if not name or len(name) < 2:
        return False
    
    return True

def merge_identification_results(results: List[Dict]) -> Dict:
    """Merge multiple identification results into a consensus"""
    if not results:
        return {'success': False, 'confidence': 0.0}
    
    # Group results by card name
    name_groups = {}
    for result in results:
        if not validate_identification_result(result):
            continue
        
        name = result['card_data'].get('name', '').lower()
        if name not in name_groups:
            name_groups[name] = []
        name_groups[name].append(result)
    
    if not name_groups:
        return {'success': False, 'confidence': 0.0}
    
    # Find consensus
    best_group = max(name_groups.values(), key=len)
    
    if len(best_group) == 1:
        return best_group[0]
    
    # Multiple results for same card - combine confidences
    total_confidence = sum(r['confidence'] for r in best_group)
    avg_confidence = total_confidence / len(best_group)
    
    best_result = max(best_group, key=lambda x: x['confidence'])
    best_result['confidence'] = min(1.0, avg_confidence * 1.2)  # Boost for consensus
    best_result['method'] = 'consensus'
    best_result['supporting_methods'] = [r['method'] for r in best_group]
    
    return best_result