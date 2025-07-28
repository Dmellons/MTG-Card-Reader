"""
Configuration settings for MTG Card Processing System
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Database Configuration
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/mtg_cards"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 30
    
    # File Storage Paths
    STORAGE_BASE_PATH: str = "./data"
    CARD_IMAGES_PATH: str = "./data/card_images"
    PDF_INPUT_PATH: str = "./data/input_pdfs"
    EXPORTS_PATH: str = "./data/exports"
    
    # Processing Settings
    PDF_DPI_SCANNED: int = 300
    PDF_DPI_TEXT: int = 400
    BATCH_SIZE: int = 1000
    
    # OCR Configuration
    OCR_ENGINE: str = "easyocr"  # Options: "easyocr", "tesseract", "both"
    OCR_CONFIDENCE_THRESHOLD: float = 0.2  # Very low threshold - let post-processing handle quality
    USE_CLOUD_OCR: bool = False
    
    # Enhanced OCR Settings
    OCR_RETRY_COUNT: int = 3
    OCR_PREPROCESSING_AGGRESSIVE: bool = True
    
    # Card Detection Settings
    EXPECTED_GRID_ROWS: int = 3
    EXPECTED_GRID_COLS: int = 3
    PERSPECTIVE_CORRECTION: bool = True
    
    # API Configuration
    SCRYFALL_API_BASE: str = "https://api.scryfall.com"
    API_RATE_LIMIT_DELAY: float = 0.1  # 100ms between requests
    API_TIMEOUT: int = 30
    
    # Image Matching
    PERCEPTUAL_HASH_THRESHOLD: int = 10
    FEATURE_MATCH_MIN_MATCHES: int = 10
    
    # Duplicate Detection
    DUPLICATE_SIMILARITY_THRESHOLD: float = 0.85
    FUZZY_MATCH_THRESHOLD: float = 0.7
    
    # Export Settings
    DEFAULT_EXPORT_FORMAT: str = "csv"  # Options: "csv", "json", "both"
    INCLUDE_IMAGES_IN_EXPORT: bool = False
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = "./logs/mtg_processor.log"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create directories if they don't exist
def ensure_directories(settings: Settings):
    """Create necessary directories"""
    directories = [
        settings.STORAGE_BASE_PATH,
        settings.CARD_IMAGES_PATH,
        settings.PDF_INPUT_PATH,
        settings.EXPORTS_PATH,
        "./logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

# Global settings instance
settings = Settings()
ensure_directories(settings)