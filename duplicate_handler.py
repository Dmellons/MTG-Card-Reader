"""
Duplicate Detection and Management Module
Handles detection and management of duplicate cards
"""
import logging
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher
from sqlalchemy.orm import Session
from models import Card, CardRepository
from config import settings

logger = logging.getLogger(__name__)

class DuplicateHandler:
    """Handles detection and management of duplicate cards"""
    
    def __init__(self):
        self.similarity_threshold = settings.DUPLICATE_SIMILARITY_THRESHOLD
        self.fuzzy_threshold = settings.FUZZY_MATCH_THRESHOLD
    
    def detect_duplicates(self, card_data: Dict, session: Session) -> Dict:
        """
        Multi-level duplicate detection
        Returns dict with duplicate status and match information
        """
        card_repo = CardRepository(session)
        
        # Level 1: Exact name + set matching
        exact_match = self._check_exact_match(card_data, card_repo)
        if exact_match:
            return {
                'is_duplicate': True,
                'type': 'exact',
                'match': exact_match,
                'confidence': 1.0,
                'reason': 'Exact name and set match'
            }
        
        # Level 2: Exact name, different/unknown set
        name_match = self._check_name_match(card_data, card_repo)
        if name_match:
            return {
                'is_duplicate': True,
                'type': 'name_only',
                'match': name_match,
                'confidence': 0.95,
                'reason': 'Exact name match, different or unknown set'
            }
        
        # Level 3: Fuzzy name matching within same set
        set_code = card_data.get('set_code')
        if set_code:
            fuzzy_set_match = self._check_fuzzy_match_in_set(card_data, card_repo, set_code)
            if fuzzy_set_match and fuzzy_set_match[1] >= self.similarity_threshold:
                return {
                    'is_duplicate': True,
                    'type': 'fuzzy_set',
                    'match': fuzzy_set_match[0],
                    'confidence': fuzzy_set_match[1],
                    'reason': f'Fuzzy name match within set {set_code}'
                }
        
        # Level 4: Fuzzy name matching across all sets
        fuzzy_match = self._check_fuzzy_match_all(card_data, card_repo)
        if fuzzy_match and fuzzy_match[1] >= self.similarity_threshold:
            return {
                'is_duplicate': True,
                'type': 'fuzzy_all',
                'match': fuzzy_match[0],
                'confidence': fuzzy_match[1],
                'reason': 'Fuzzy name match across all sets'
            }
        
        return {
            'is_duplicate': False,
            'type': None,
            'match': None,
            'confidence': 0.0,
            'reason': 'No duplicates found'
        }
    
    def _check_exact_match(self, card_data: Dict, card_repo: CardRepository) -> Optional[Card]:
        """Check for exact name and set match"""
        name = card_data.get('name')
        set_code = card_data.get('set_code')
        
        if not name:
            return None
        
        # Try with set code if available
        if set_code:
            match = card_repo.find_by_name_and_set(name, set_code)
            if match:
                logger.debug(f"Exact match found: {name} in {set_code}")
                return match
        
        return None
    
    def _check_name_match(self, card_data: Dict, card_repo: CardRepository) -> Optional[Card]:
        """Check for exact name match regardless of set"""
        name = card_data.get('name')
        if not name:
            return None
        
        # Search for any card with exact name
        similar_cards = card_repo.find_similar_cards(name, limit=1)
        
        for card in similar_cards:
            if (card.name and name and 
                isinstance(card.name, str) and isinstance(name, str) and
                card.name.lower() == name.lower()):
                logger.debug(f"Name-only match found: {name}")
                return card
        
        return None
    
    def _check_fuzzy_match_in_set(self, card_data: Dict, card_repo: CardRepository, 
                                 set_code: str) -> Optional[Tuple[Card, float]]:
        """Check for fuzzy name match within specific set"""
        name = card_data.get('name')
        if not name or len(name) < 3:
            return None
        
        candidates = card_repo.find_similar_cards(name, set_code, limit=10)
        return self._find_best_fuzzy_match(name, candidates)
    
    def _check_fuzzy_match_all(self, card_data: Dict, card_repo: CardRepository) -> Optional[Tuple[Card, float]]:
        """Check for fuzzy name match across all sets"""
        name = card_data.get('name')
        if not name or len(name) < 3:
            return None
        
        candidates = card_repo.find_similar_cards(name, limit=20)
        return self._find_best_fuzzy_match(name, candidates)
    
    def _find_best_fuzzy_match(self, query_name: str, candidates: List[Card]) -> Optional[Tuple[Card, float]]:
        """Find the best fuzzy match from a list of candidates"""
        if not candidates:
            return None
        
        query_normalized = self._normalize_name(query_name)
        best_match = None
        best_similarity = 0.0
        
        for candidate in candidates:
            if not candidate.name:
                continue
            candidate_normalized = self._normalize_name(candidate.name)
            
            # Calculate multiple similarity metrics
            similarities = [
                SequenceMatcher(None, query_normalized, candidate_normalized).ratio(),
                self._jaro_winkler_similarity(query_normalized, candidate_normalized),
                self._levenshtein_similarity(query_normalized, candidate_normalized)
            ]
            
            # Use the maximum similarity
            similarity = max(similarities)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = candidate
        
        if best_similarity >= self.fuzzy_threshold:
            logger.debug(f"Fuzzy match: '{query_name}' -> '{best_match.name}' "
                        f"(similarity: {best_similarity:.3f})")
            return (best_match, best_similarity)
        
        return None
    
    def _normalize_name(self, name: str) -> str:
        """Normalize card name for better matching"""
        if not name or not isinstance(name, str):
            return ""
        
        # Convert to lowercase and remove extra whitespace
        normalized = name.lower().strip()
        
        # Remove punctuation and special characters
        import string
        normalized = ''.join(char for char in normalized if char.isalnum() or char.isspace())
        
        # Normalize multiple spaces to single space
        normalized = ' '.join(normalized.split())
        
        # Handle common OCR errors and variations
        ocr_corrections = {
            'ae': 'æ',
            'oe': 'œ',
            # Numbers that might be misread as letters
            '0': 'o',
            '1': 'l', '1': 'i',
            '5': 's',
            '8': 'b',
            # Common article variations
            'the ': '',  # Remove articles for better matching
            'a ': '',
            'an ': ''
        }
        
        for error, correction in ocr_corrections.items():
            normalized = normalized.replace(error, correction)
        
        return normalized
    
    def _jaro_winkler_similarity(self, s1: str, s2: str) -> float:
        """Calculate Jaro-Winkler similarity"""
        try:
            # Simple implementation - in production, use a proper library
            if s1 == s2:
                return 1.0
            
            len1, len2 = len(s1), len(s2)
            if len1 == 0 or len2 == 0:
                return 0.0
            
            match_window = max(len1, len2) // 2 - 1
            if match_window < 0:
                match_window = 0
            
            s1_matches = [False] * len1
            s2_matches = [False] * len2
            
            matches = 0
            transpositions = 0
            
            # Identify matches
            for i in range(len1):
                start = max(0, i - match_window)
                end = min(i + match_window + 1, len2)
                
                for j in range(start, end):
                    if s2_matches[j] or s1[i] != s2[j]:
                        continue
                    s1_matches[i] = s2_matches[j] = True
                    matches += 1
                    break
            
            if matches == 0:
                return 0.0
            
            # Count transpositions
            k = 0
            for i in range(len1):
                if not s1_matches[i]:
                    continue
                while not s2_matches[k]:
                    k += 1
                if s1[i] != s2[k]:
                    transpositions += 1
                k += 1
            
            # Calculate Jaro similarity
            jaro = (matches / len1 + matches / len2 + 
                   (matches - transpositions / 2) / matches) / 3
            
            # Calculate Jaro-Winkler similarity
            prefix = 0
            for i in range(min(len1, len2, 4)):
                if s1[i] == s2[i]:
                    prefix += 1
                else:
                    break
            
            return jaro + 0.1 * prefix * (1 - jaro)
        
        except:
            # Fallback to simple ratio if calculation fails
            return SequenceMatcher(None, s1, s2).ratio()
    
    def _levenshtein_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity based on Levenshtein distance"""
        try:
            len1, len2 = len(s1), len(s2)
            if len1 == 0 or len2 == 0:
                return 0.0
            
            # Create distance matrix
            matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]
            
            # Initialize first row and column
            for i in range(len1 + 1):
                matrix[i][0] = i
            for j in range(len2 + 1):
                matrix[0][j] = j
            
            # Fill matrix
            for i in range(1, len1 + 1):
                for j in range(1, len2 + 1):
                    cost = 0 if s1[i-1] == s2[j-1] else 1
                    matrix[i][j] = min(
                        matrix[i-1][j] + 1,      # deletion
                        matrix[i][j-1] + 1,      # insertion
                        matrix[i-1][j-1] + cost  # substitution
                    )
            
            # Convert distance to similarity
            distance = matrix[len1][len2]
            max_len = max(len1, len2)
            
            return 1.0 - (distance / max_len) if max_len > 0 else 0.0
        
        except:
            return SequenceMatcher(None, s1, s2).ratio()

class DuplicateManager:
    """Manages duplicate relationships and resolution"""
    
    def __init__(self, session: Session):
        self.session = session
        self.card_repo = CardRepository(session)
    
    def get_duplicate_groups(self, collection_name: str = None) -> List[Dict]:
        """Get all duplicate groups, optionally filtered by collection"""
        from sqlalchemy import and_
        from models import CardDuplicate
        
        query = self.session.query(CardDuplicate)
        
        if collection_name:
            query = query.join(Card, CardDuplicate.original_card_id == Card.id)
            query = query.filter(Card.collection_name == collection_name)
        
        duplicates = query.all()
        
        # Group duplicates by original card
        groups = {}
        for dup in duplicates:
            original_id = str(dup.original_card_id)
            if original_id not in groups:
                groups[original_id] = {
                    'original_card': self._card_to_dict(dup.original_card),
                    'duplicates': []
                }
            
            groups[original_id]['duplicates'].append({
                'card': self._card_to_dict(dup.duplicate_card),
                'similarity_score': dup.similarity_score,
                'detection_method': dup.detection_method,
                'created_date': dup.created_date.isoformat()
            })
        
        return list(groups.values())
    
    def resolve_duplicate(self, duplicate_id: str, action: str, 
                         keep_card_id: str = None) -> Dict:
        """
        Resolve a duplicate relationship
        Actions: 'confirm', 'reject', 'merge'
        """
        from models import CardDuplicate
        
        try:
            duplicate = self.session.query(CardDuplicate).filter(
                CardDuplicate.id == duplicate_id
            ).first()
            
            if not duplicate:
                return {'success': False, 'error': 'Duplicate not found'}
            
            if action == 'confirm':
                # Mark as confirmed duplicate - no action needed
                return {'success': True, 'action': 'confirmed'}
            
            elif action == 'reject':
                # Remove duplicate relationship
                self.session.delete(duplicate)
                self.session.commit()
                return {'success': True, 'action': 'rejected'}
            
            elif action == 'merge':
                # Merge duplicate cards (keep one, mark other as merged)
                if not keep_card_id:
                    return {'success': False, 'error': 'keep_card_id required for merge'}
                
                # Implementation would depend on business rules
                # For now, just remove the duplicate relationship
                self.session.delete(duplicate)
                self.session.commit()
                return {'success': True, 'action': 'merged'}
            
            else:
                return {'success': False, 'error': f'Unknown action: {action}'}
        
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error resolving duplicate: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_duplicate_statistics(self, collection_name: str = None) -> Dict:
        """Get statistics about duplicates"""
        from sqlalchemy import func
        from models import CardDuplicate
        
        query = self.session.query(CardDuplicate)
        
        if collection_name:
            query = query.join(Card, CardDuplicate.original_card_id == Card.id)
            query = query.filter(Card.collection_name == collection_name)
        
        total_duplicates = query.count()
        
        # Group by detection method
        method_stats = dict(query.with_entities(
            CardDuplicate.detection_method,
            func.count(CardDuplicate.id)
        ).group_by(CardDuplicate.detection_method).all())
        
        # Average similarity score
        avg_similarity = query.with_entities(
            func.avg(CardDuplicate.similarity_score)
        ).scalar() or 0.0
        
        return {
            'total_duplicates': total_duplicates,
            'method_breakdown': method_stats,
            'average_similarity': float(avg_similarity),
            'collection_name': collection_name
        }
    
    def _card_to_dict(self, card: Card) -> Dict:
        """Convert Card model to dictionary"""
        return {
            'id': str(card.id),
            'name': card.name,
            'set_code': card.set_code,
            'collection_name': card.collection_name,
            'confidence_score': card.confidence_score,
            'identification_method': card.identification_method
        }

class DuplicateAnalyzer:
    """Analyzes duplicate patterns and provides insights"""
    
    def __init__(self, session: Session):
        self.session = session
        self.duplicate_manager = DuplicateManager(session)
    
    def analyze_duplicate_patterns(self, collection_name: str = None) -> Dict:
        """Analyze patterns in duplicate detection"""
        stats = self.duplicate_manager.get_duplicate_statistics(collection_name)
        groups = self.duplicate_manager.get_duplicate_groups(collection_name)
        
        # Analyze common duplicate causes
        causes = {
            'ocr_errors': 0,
            'set_variations': 0,
            'image_quality': 0,
            'legitimate_reprints': 0
        }
        
        for group in groups:
            original_name = group['original_card']['name']
            
            for dup in group['duplicates']:
                dup_name = dup['card']['name']
                similarity = dup['similarity_score']
                
                # Classify duplicate cause
                if (similarity > 0.9 and original_name and dup_name and 
                    isinstance(original_name, str) and isinstance(dup_name, str) and
                    original_name.lower() != dup_name.lower()):
                    causes['ocr_errors'] += 1
                elif (original_name and dup_name and 
                      isinstance(original_name, str) and isinstance(dup_name, str) and
                      original_name.lower() == dup_name.lower()):
                    causes['legitimate_reprints'] += 1
                elif 0.7 <= similarity <= 0.9:
                    causes['image_quality'] += 1
                else:
                    causes['set_variations'] += 1
        
        return {
            'statistics': stats,
            'duplicate_causes': causes,
            'total_groups': len(groups),
            'recommendations': self._generate_recommendations(causes, stats)
        }
    
    def _generate_recommendations(self, causes: Dict, stats: Dict) -> List[str]:
        """Generate recommendations based on duplicate analysis"""
        recommendations = []
        
        total_duplicates = stats.get('total_duplicates', 0)
        if total_duplicates == 0:
            return ['No duplicates detected - excellent data quality!']
        
        ocr_ratio = causes['ocr_errors'] / total_duplicates if total_duplicates > 0 else 0
        if ocr_ratio > 0.3:
            recommendations.append(
                'High OCR error rate detected. Consider improving image quality or OCR preprocessing.'
            )
        
        reprint_ratio = causes['legitimate_reprints'] / total_duplicates if total_duplicates > 0 else 0
        if reprint_ratio > 0.5:
            recommendations.append(
                'Many legitimate reprints detected. Consider adjusting duplicate detection sensitivity.'
            )
        
        avg_similarity = stats.get('average_similarity', 0)
        if avg_similarity < 0.8:
            recommendations.append(
                'Low average similarity suggests aggressive duplicate detection. Consider raising thresholds.'
            )
        
        if not recommendations:
            recommendations.append('Duplicate detection appears to be working well.')
        
        return recommendations