# MTG Card Processing System

A comprehensive Python system for processing PDF collections of Magic: The Gathering cards, extracting structured data, and managing card databases.

## Features

- **PDF Processing**: Handles both scanned and text-based PDFs
- **Card Detection**: Automatically segments individual cards from grid layouts
- **OCR Text Extraction**: Extracts card names, mana costs, rules text, and more
- **Card Identification**: Uses Scryfall API and image matching for accurate identification
- **Duplicate Detection**: Multi-level duplicate detection with configurable thresholds
- **Database Management**: PostgreSQL backend with comprehensive data modeling
- **Export Capabilities**: CSV, JSON, and database dump exports
- **CLI Interface**: Easy-to-use command-line interface

## Installation

### Prerequisites

- Python 3.8 or higher
- PostgreSQL database
- Tesseract OCR (optional, for fallback OCR)

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd mtg-card-processor
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure environment**:
```bash
cp .env.template .env
# Edit .env with your database credentials and preferences
```

4. **Initialize database**:
```bash
python cli.py database init
```

## Quick Start

### Process a PDF Collection

```bash
# Process a single PDF
python cli.py process cards.pdf --collection "My Collection"

# Process with custom export formats
python cli.py process cards.pdf -c "Vintage Cards" -f csv -f json -f report

# Batch process a directory
python cli.py batch /path/to/pdfs --collection-prefix "batch_2024"
```

### Search and Export

```bash
# Search for cards
python cli.py database search "Lightning Bolt" --limit 5

# Export a collection
python cli.py export collection "My Collection" --format csv

# Generate analysis report
python cli.py export report "My Collection" --detailed
```

### Manage Duplicates

```bash
# List duplicates
python cli.py duplicates list --collection "My Collection"

# Analyze duplicate patterns
python cli.py duplicates analyze
```

## System Architecture

### Core Components

1. **PDF Processor** (`pdf_processor.py`)
   - Handles PDF analysis and image extraction
   - Supports both scanned and text-based PDFs
   - Automatic perspective correction

2. **Card Detector** (`card_detector.py`)
   - Grid layout detection using computer vision
   - Contour-based fallback detection
   - Configurable grid parameters

3. **OCR Extractor** (`ocr_extractor.py`)
   - Multi-engine OCR support (EasyOCR, Tesseract)
   - MTG-specific text parsing
   - Confidence scoring and validation

4. **Card Identifier** (`card_identifier.py`)
   - Scryfall API integration
   - Image hash matching
   - Fuzzy name matching
   - Hybrid identification approach

5. **Duplicate Handler** (`duplicate_handler.py`)
   - Multi-level duplicate detection
   - Similarity algorithms
   - Duplicate relationship management

6. **Export Manager** (`export_manager.py`)
   - Multiple export formats
   - Collection analytics
   - Batch export capabilities

### Database Schema

The system uses PostgreSQL with the following main tables:

- **cards**: Main card data storage
- **card_duplicates**: Duplicate relationships
- **processing_logs**: Processing history and errors
- **collections**: Collection metadata

## Configuration

Key configuration options in `.env`:

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/mtg_cards

# Processing
OCR_ENGINE=easyocr  # or tesseract, both
EXPECTED_GRID_ROWS=3
EXPECTED_GRID_COLS=3

# Quality thresholds
DUPLICATE_SIMILARITY_THRESHOLD=0.85
OCR_CONFIDENCE_THRESHOLD=0.5
```

## Usage Examples

### Python API

```python
from main_processor import MTGCardProcessingSystem

# Initialize system
processor = MTGCardProcessingSystem()

# Process a PDF
result = processor.process_pdf_collection(
    "cards.pdf", 
    collection_name="My Collection"
)

# Search cards
cards = processor.search_cards("Lightning", {"rarity": "common"})

# Get statistics
stats = processor.get_collection_statistics("My Collection")
```

### Advanced Processing

```python
# Process single card image
result = processor.process_single_card_image("card.jpg")

# Custom card detection
from card_detector import CardDetector
detector = CardDetector()
cards = detector.detect_card_grid_advanced(image, rows=4, cols=4)

# Export with custom filters
from export_manager import ExportManager
exporter = ExportManager()
result = exporter.export_collection_to_csv(
    "My Collection", 
    "custom_export.csv"
)
```

## Performance Optimization

### Processing Speed
- Use appropriate DPI settings (300 for scanned, 400 for text PDFs)
- Enable batch processing for large collections
- Configure database connection pooling

### Memory Management
- Process large PDFs in smaller batches
- Use streaming database queries
- Clean up temporary files regularly

### Accuracy Improvements
- Ensure high-quality source PDFs
- Adjust OCR confidence thresholds
- Use perspective correction for scanned documents
- Build image hash databases for faster matching

## Troubleshooting

### Common Issues

1. **OCR Recognition Errors**
   - Increase image DPI settings
   - Enable perspective correction
   - Try different OCR engines

2. **Card Detection Problems**
   - Adjust grid parameters
   - Use manual grid specification
   - Enable debug logging

3. **Database Connection Issues**
   - Verify DATABASE_URL format
   - Check PostgreSQL service status
   - Adjust connection pool settings

4. **API Rate Limiting**
   - Increase API_RATE_LIMIT_DELAY
   - Use bulk data downloads for large datasets

### Debug Mode

Enable verbose logging:
```bash
python cli.py -v process cards.pdf
```

Check log files:
```bash
tail -f logs/mtg_processor.log
```

## Data Quality

### Confidence Scoring
The system provides confidence scores for:
- OCR extraction quality
- Card identification accuracy
- Overall processing reliability

### Validation
- Card name format validation
- Mana cost syntax checking
- Power/toughness validation for creatures
- Set code verification

### Quality Reports
Generate quality reports to identify:
- Low confidence extractions
- Processing errors
- Duplicate patterns
- Collection completeness

## API Integration

### Scryfall API
- Automatic rate limiting compliance
- Fuzzy name matching
- Bulk data support
- Comprehensive card data

### Image Matching
- Perceptual hash comparison
- Feature-based matching (ORB/SIFT)
- Similarity thresholds
- Reference image databases

## Contributing

### Development Setup
1. Install development dependencies:
```bash
pip install -r requirements.txt
```

2. Run tests:
```bash
pytest tests/
```

3. Code formatting:
```bash
black . && flake8 .
```

### Adding Features
- Follow the modular architecture
- Add comprehensive tests
- Update documentation
- Use type hints
- Follow PEP 8 style guidelines

## License

This project is for educational purposes. Magic: The Gathering is a trademark of Wizards of the Coast LLC.

## Acknowledgments

- [Scryfall](https://scryfall.com/) for their excellent MTG API
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for OCR capabilities
- [OpenCV](https://opencv.org/) for computer vision tools
- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF processing

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review log files for errors
3. Consult the configuration documentation
4. Create an issue with detailed information