"""
Database models for MTG Card Processing System
"""
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Float, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class Card(Base):
    """Main card table storing all card information"""
    __tablename__ = 'cards'
    
    # Primary identifiers
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    scryfall_id = Column(UUID(as_uuid=True), unique=True, nullable=True)
    
    # Card attributes
    mana_cost = Column(String(100))
    converted_mana_cost = Column(Integer, default=0)
    type_line = Column(String(255))
    oracle_text = Column(Text)
    flavor_text = Column(Text)
    power = Column(String(10))  # Can be '*' or number
    toughness = Column(String(10))  # Can be '*' or number
    
    # Set and rarity information
    set_code = Column(String(10), nullable=False, index=True)
    set_name = Column(String(255))
    rarity = Column(String(20), index=True)
    collector_number = Column(String(20))
    
    # Color information
    colors = Column(String(20))  # e.g., "WU" for white/blue
    color_identity = Column(String(20))
    
    # Processing metadata
    image_path = Column(String(500))
    source_pdf = Column(String(500))
    source_page = Column(Integer)
    card_position = Column(Integer)  # Position on page
    processed_date = Column(DateTime, default=datetime.utcnow)
    confidence_score = Column(Float)  # OCR/identification confidence
    identification_method = Column(String(50))  # 'ocr_api', 'image_hash', 'manual'
    
    # Collection information
    collection_name = Column(String(255), index=True)
    
    # Relationships
    duplicates = relationship("CardDuplicate", foreign_keys="CardDuplicate.original_card_id", back_populates="original_card")
    duplicate_of = relationship("CardDuplicate", foreign_keys="CardDuplicate.duplicate_card_id", back_populates="duplicate_card")

class CardDuplicate(Base):
    """Track duplicate cards found during processing"""
    __tablename__ = 'card_duplicates'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    original_card_id = Column(UUID(as_uuid=True), ForeignKey('cards.id'), nullable=False)
    duplicate_card_id = Column(UUID(as_uuid=True), ForeignKey('cards.id'), nullable=False)
    similarity_score = Column(Float, nullable=False)
    detection_method = Column(String(50))  # 'exact', 'fuzzy', 'image_hash'
    created_date = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    original_card = relationship("Card", foreign_keys=[original_card_id])
    duplicate_card = relationship("Card", foreign_keys=[duplicate_card_id])

class ProcessingLog(Base):
    """Log processing activities and errors"""
    __tablename__ = 'processing_logs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow)
    level = Column(String(20))  # INFO, WARNING, ERROR
    message = Column(Text)
    source_file = Column(String(500))
    card_id = Column(UUID(as_uuid=True), ForeignKey('cards.id'), nullable=True)
    error_details = Column(Text)  # JSON string for structured error info
    
    # Relationship
    card = relationship("Card")

class Collection(Base):
    """Store collection metadata"""
    __tablename__ = 'collections'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), unique=True, nullable=False)
    description = Column(Text)
    created_date = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow)
    total_cards = Column(Integer, default=0)
    unique_cards = Column(Integer, default=0)
    source_files = Column(Text)  # JSON array of source PDF files

class DatabaseManager:
    """Database connection and session management"""
    
    def __init__(self, database_url: str, pool_size: int = 20, max_overflow: int = 30):
        self.engine = create_engine(
            database_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,  # Verify connections before use
            echo=False  # Set to True for SQL debugging
        )
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get a new database session"""
        return self.SessionLocal()
    
    def drop_all_tables(self):
        """Drop all tables (use with caution!)"""
        Base.metadata.drop_all(bind=self.engine)

class CardRepository:
    """Repository pattern for card database operations"""
    
    def __init__(self, session):
        self.session = session
    
    def create_card(self, card_data: dict) -> Card:
        """Create a new card record"""
        card = Card(**card_data)
        self.session.add(card)
        self.session.flush()  # Get the ID without committing
        return card
    
    def find_by_name_and_set(self, name: str, set_code: str) -> Card:
        """Find card by exact name and set match"""
        return self.session.query(Card).filter(
            Card.name == name,
            Card.set_code == set_code
        ).first()
    
    def find_similar_cards(self, name: str, set_code: str = None, limit: int = 10):
        """Find similar cards using database search"""
        query = self.session.query(Card)
        
        if set_code:
            query = query.filter(Card.set_code == set_code)
        
        # Use PostgreSQL similarity search if available
        # For now, use simple LIKE matching
        query = query.filter(Card.name.ilike(f'%{name}%'))
        
        return query.limit(limit).all()
    
    def get_collection_stats(self, collection_name: str = None):
        """Get statistics for a collection"""
        from sqlalchemy import func
        
        query = self.session.query(Card)
        if collection_name:
            query = query.filter(Card.collection_name == collection_name)
        
        total_cards = query.count()
        unique_cards = query.distinct(Card.name, Card.set_code).count()
        
        # Rarity distribution
        rarity_dist = dict(query.with_entities(
            Card.rarity, func.count(Card.id)
        ).group_by(Card.rarity).all())
        
        # Set distribution
        set_dist = dict(query.with_entities(
            Card.set_code, func.count(Card.id)
        ).group_by(Card.set_code).all())
        
        return {
            'total_cards': total_cards,
            'unique_cards': unique_cards,
            'rarity_distribution': rarity_dist,
            'set_distribution': set_dist
        }
    
    def bulk_create_cards(self, cards_data: list) -> int:
        """Bulk create multiple cards efficiently"""
        cards = [Card(**card_data) for card_data in cards_data]
        self.session.bulk_save_objects(cards)
        return len(cards)
    
    def mark_as_duplicate(self, original_card: Card, duplicate_card: Card, 
                         similarity_score: float, method: str):
        """Mark a card as duplicate of another"""
        duplicate_record = CardDuplicate(
            original_card_id=original_card.id,
            duplicate_card_id=duplicate_card.id,
            similarity_score=similarity_score,
            detection_method=method
        )
        self.session.add(duplicate_record)
    
    def get_cards_by_collection(self, collection_name: str, limit: int = None):
        """Get all cards in a specific collection"""
        query = self.session.query(Card).filter(
            Card.collection_name == collection_name
        ).order_by(Card.name)
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    def search_cards(self, search_term: str, filters: dict = None):
        """Search cards with various filters"""
        query = self.session.query(Card)
        
        # Text search in name and oracle text
        if search_term:
            search_filter = Card.name.ilike(f'%{search_term}%')
            if hasattr(Card, 'oracle_text'):
                search_filter = search_filter | Card.oracle_text.ilike(f'%{search_term}%')
            query = query.filter(search_filter)
        
        # Apply additional filters
        if filters:
            if 'set_code' in filters:
                query = query.filter(Card.set_code == filters['set_code'])
            if 'rarity' in filters:
                query = query.filter(Card.rarity == filters['rarity'])
            if 'colors' in filters:
                query = query.filter(Card.colors.like(f'%{filters["colors"]}%'))
            if 'collection' in filters:
                query = query.filter(Card.collection_name == filters['collection'])
        
        return query.all()