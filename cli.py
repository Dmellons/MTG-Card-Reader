#!/usr/bin/env python3
"""
Command Line Interface for MTG Card Processing System
Provides easy command-line access to all system functions
"""
import os
import sys
import click
import json
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main_processor import MTGCardProcessingSystem, BatchProcessor, setup_processing_directories, cleanup_old_files
from export_manager import ExportManager, BulkExporter
from duplicate_handler import DuplicateManager, DuplicateAnalyzer
from models import DatabaseManager
from config import settings

@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', '-c', help='Path to configuration file')
def cli(verbose, config):
    """MTG Card Processing System CLI
    
    Process PDF collections of Magic: The Gathering cards and extract structured data.
    """
    if verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    if config:
        # Load custom configuration if provided
        click.echo(f"Loading configuration from: {config}")
    
    # Ensure directories exist
    setup_processing_directories()

@cli.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--collection', '-c', help='Collection name for the processed cards')
@click.option('--output-dir', '-o', help='Output directory for exports', default=None)
@click.option('--export-format', '-f', multiple=True, 
              type=click.Choice(['csv', 'json', 'report', 'all']),
              default=['csv'], help='Export formats to generate')
@click.option('--skip-duplicates', is_flag=True, help='Skip duplicate detection')
def process(pdf_path, collection, output_dir, export_format, skip_duplicates):
    """Process a PDF file containing MTG cards"""
    
    click.echo(f"🎯 Processing PDF: {pdf_path}")
    
    try:
        # Initialize processor
        processor = MTGCardProcessingSystem()
        
        # Generate collection name if not provided
        if not collection:
            filename = Path(pdf_path).stem
            collection = f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Process the PDF
        with click.progressbar(length=100, label='Processing PDF') as bar:
            result = processor.process_pdf_collection(pdf_path, collection)
            bar.update(100)
        
        if result.get('success'):
            click.echo("✅ Processing completed successfully!")
            click.echo(f"📊 Results:")
            click.echo(f"   • Pages processed: {result.get('pages_processed', 0)}")
            click.echo(f"   • Cards detected: {result.get('cards_detected', 0)}")
            click.echo(f"   • Cards stored: {result.get('cards_stored', 0)}")
            click.echo(f"   • Duplicates found: {result.get('duplicates_found', 0)}")
            click.echo(f"   • Processing time: {result.get('processing_time', 0):.2f} seconds")
            
            # Generate exports
            if 'all' in export_format:
                export_format = ['csv', 'json', 'report']
            
            if export_format:
                click.echo("\n📤 Generating exports...")
                export_manager = ExportManager()
                
                for fmt in export_format:
                    if fmt == 'csv':
                        export_result = export_manager.export_collection_to_csv(collection)
                    elif fmt == 'json':
                        export_result = export_manager.export_collection_to_json(collection)
                    elif fmt == 'report':
                        export_result = export_manager.generate_collection_report(collection)
                    
                    if export_result.get('success'):
                        click.echo(f"   ✅ {fmt.upper()}: {export_result['file_path']}")
                    else:
                        click.echo(f"   ❌ {fmt.upper()}: {export_result.get('error', 'Unknown error')}")
        else:
            click.echo(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    
    except Exception as e:
        click.echo(f"Fatal error: {str(e)}")
        sys.exit(1)

@cli.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False))
@click.option('--collection-prefix', '-p', default='batch', 
              help='Prefix for collection names')
@click.option('--parallel', is_flag=True, help='Process files in parallel (experimental)')
def batch(directory, collection_prefix, parallel):
    """Process all PDF files in a directory"""
    
    click.echo(f"Processing directory: {directory}")
    
    # Find PDF files
    pdf_files = list(Path(directory).glob("*.pdf"))
    
    if not pdf_files:
        click.echo("No PDF files found in directory")
        sys.exit(1)
    
    click.echo(f"Found {len(pdf_files)} PDF files")
    
    try:
        processor = MTGCardProcessingSystem()
        batch_processor = BatchProcessor(processor)
        
        with click.progressbar(pdf_files, label='Processing PDFs') as bar:
            result = batch_processor.process_pdf_directory(directory, collection_prefix)
            bar.update(len(pdf_files))
        
        if result.get('success'):
            click.echo("Batch processing completed!")
            click.echo(f"📊 Results:")
            click.echo(f"   • Files processed: {result.get('files_processed', 0)}")
            click.echo(f"   • Total cards stored: {result.get('total_cards_stored', 0)}")
            click.echo(f"   • Total duplicates: {result.get('total_duplicates', 0)}")
        else:
            click.echo(f"❌ Batch processing failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        click.echo(f"Fatal error: {str(e)}")
        sys.exit(1)

@cli.group()
def export():
    """Export and reporting commands"""
    pass

@export.command('collection')
@click.argument('collection_name')
@click.option('--format', '-f', type=click.Choice(['csv', 'json', 'sql', 'sqlite']),
              default='csv', help='Export format')
@click.option('--output', '-o', help='Output file path')
def export_collection(collection_name, format, output):
    """Export a specific collection"""
    
    click.echo(f"📤 Exporting collection: {collection_name}")
    
    try:
        export_manager = ExportManager()
        
        if format == 'csv':
            result = export_manager.export_collection_to_csv(collection_name, output)
        elif format == 'json':
            result = export_manager.export_collection_to_json(collection_name, output)
        elif format in ['sql', 'sqlite']:
            result = export_manager.export_to_database_dump(collection_name, format)
        
        if result.get('success'):
            click.echo(f"✅ Export completed: {result['file_path']}")
            click.echo(f"📊 Cards exported: {result.get('cards_exported', 0)}")
        else:
            click.echo(f"❌ Export failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        click.echo(f"💥 Export error: {str(e)}")

@export.command('all')
@click.option('--format', '-f', multiple=True,
              type=click.Choice(['csv', 'json', 'report']),
              default=['csv'], help='Export formats')
def export_all(format):
    """Export all collections"""
    
    click.echo("📤 Exporting all collections...")
    
    try:
        bulk_exporter = BulkExporter()
        result = bulk_exporter.export_all_collections(list(format))
        
        if result.get('success'):
            click.echo(f"✅ Bulk export completed!")
            click.echo(f"📊 Collections processed: {result.get('collections_processed', 0)}")
            
            # Show results for each collection
            for collection, formats in result.get('results', {}).items():
                click.echo(f"\n📁 {collection}:")
                for fmt, fmt_result in formats.items():
                    if fmt_result.get('success'):
                        click.echo(f"   ✅ {fmt.upper()}: {fmt_result.get('file_path', 'Generated')}")
                    else:
                        click.echo(f"   ❌ {fmt.upper()}: {fmt_result.get('error', 'Failed')}")
        else:
            click.echo(f"❌ Bulk export failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        click.echo(f"💥 Export error: {str(e)}")

@export.command('report')
@click.argument('collection_name', required=False)
@click.option('--output', '-o', help='Output file path')
@click.option('--detailed', is_flag=True, help='Include detailed analytics')
def export_report(collection_name, output, detailed):
    """Generate collection analysis report"""
    
    if collection_name:
        click.echo(f"📊 Generating report for: {collection_name}")
    else:
        click.echo("📊 Generating report for all collections")
    
    try:
        export_manager = ExportManager()
        result = export_manager.generate_collection_report(collection_name, output)
        
        if result.get('success'):
            click.echo(f"✅ Report generated: {result['file_path']}")
            
            # Show summary
            summary = result.get('report_summary', {})
            click.echo(f"\n📈 Summary:")
            click.echo(f"   • Total cards: {summary.get('total_cards', 0)}")
            click.echo(f"   • Unique cards: {summary.get('unique_cards', 0)}")
            click.echo(f"   • Duplicate ratio: {summary.get('duplicate_ratio', 0):.1f}%")
        else:
            click.echo(f"❌ Report generation failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        click.echo(f"💥 Report error: {str(e)}")

@cli.group()
def duplicates():
    """Duplicate management commands"""
    pass

@duplicates.command('list')
@click.option('--collection', '-c', help='Filter by collection name')
@click.option('--method', '-m', help='Filter by detection method')
@click.option('--min-similarity', type=float, default=0.7, 
              help='Minimum similarity threshold')
def list_duplicates(collection, method, min_similarity):
    """List detected duplicates"""
    
    click.echo("🔍 Searching for duplicates...")
    
    try:
        db_manager = DatabaseManager(settings.DATABASE_URL)
        session = db_manager.get_session()
        duplicate_manager = DuplicateManager(session)
        
        groups = duplicate_manager.get_duplicate_groups(collection)
        
        if not groups:
            click.echo("✅ No duplicates found!")
            return
        
        # Filter by similarity if specified
        if min_similarity > 0:
            filtered_groups = []
            for group in groups:
                filtered_dups = [d for d in group['duplicates'] 
                               if d['similarity_score'] >= min_similarity]
                if filtered_dups:
                    group['duplicates'] = filtered_dups
                    filtered_groups.append(group)
            groups = filtered_groups
        
        click.echo(f"🎯 Found {len(groups)} duplicate groups:")
        
        for i, group in enumerate(groups, 1):
            original = group['original_card']
            click.echo(f"\n{i}. Original: {original['name']} ({original['set_code']})")
            
            for dup in group['duplicates']:
                card = dup['card']
                similarity = dup['similarity_score']
                method = dup['detection_method']
                
                click.echo(f"   └─ {card['name']} ({card['set_code']}) "
                          f"- {similarity:.2f} similarity ({method})")
        
        session.close()
    
    except Exception as e:
        click.echo(f"💥 Error listing duplicates: {str(e)}")

@duplicates.command('analyze')
@click.option('--collection', '-c', help='Analyze specific collection')
def analyze_duplicates(collection):
    """Analyze duplicate patterns and quality"""
    
    if collection:
        click.echo(f"🔬 Analyzing duplicates in: {collection}")
    else:
        click.echo("🔬 Analyzing duplicates across all collections")
    
    try:
        db_manager = DatabaseManager(settings.DATABASE_URL)
        session = db_manager.get_session()
        analyzer = DuplicateAnalyzer(session)
        
        analysis = analyzer.analyze_duplicate_patterns(collection)
        
        stats = analysis['statistics']
        causes = analysis['duplicate_causes']
        
        click.echo(f"\n📊 Duplicate Statistics:")
        click.echo(f"   • Total duplicates: {stats['total_duplicates']}")
        click.echo(f"   • Average similarity: {stats['average_similarity']:.2f}")
        click.echo(f"   • Duplicate groups: {analysis['total_groups']}")
        
        click.echo(f"\n🔍 Causes Analysis:")
        click.echo(f"   • OCR errors: {causes['ocr_errors']}")
        click.echo(f"   • Set variations: {causes['set_variations']}")
        click.echo(f"   • Image quality: {causes['image_quality']}")
        click.echo(f"   • Legitimate reprints: {causes['legitimate_reprints']}")
        
        click.echo(f"\n💡 Recommendations:")
        for rec in analysis['recommendations']:
            click.echo(f"   • {rec}")
        
        session.close()
    
    except Exception as e:
        click.echo(f"💥 Analysis error: {str(e)}")

@cli.group()
def database():
    """Database management commands"""
    pass

@database.command('init')
@click.option('--reset', is_flag=True, help='Reset existing database')
def init_database(reset):
    """Initialize the database"""
    
    click.echo("🗄️  Initializing database...")
    
    try:
        db_manager = DatabaseManager(settings.DATABASE_URL)
        
        if reset:
            if click.confirm("⚠️  This will delete all existing data. Continue?"):
                db_manager.drop_all_tables()
                click.echo("🗑️  Existing tables dropped")
        
        db_manager.create_tables()
        click.echo("✅ Database initialized successfully!")
    
    except Exception as e:
        click.echo(f"💥 Database initialization failed: {str(e)}")

@database.command('stats')
@click.option('--collection', '-c', help='Show stats for specific collection')
def database_stats(collection):
    """Show database statistics"""
    
    try:
        processor = MTGCardProcessingSystem()
        stats = processor.get_collection_statistics(collection)
        
        if collection:
            click.echo(f"📊 Statistics for collection: {collection}")
        else:
            click.echo("📊 Database Statistics")
        
        click.echo(f"\n📈 Overview:")
        click.echo(f"   • Total cards: {stats['total_cards']}")
        click.echo(f"   • Unique cards: {stats['unique_cards']}")
        
        click.echo(f"\n🎭 Rarity Distribution:")
        for rarity, count in stats['rarity_distribution'].items():
            click.echo(f"   • {rarity or 'Unknown'}: {count}")
        
        click.echo(f"\n📦 Set Distribution:")
        set_items = sorted(stats['set_distribution'].items(), key=lambda x: x[1], reverse=True)
        for set_code, count in set_items[:10]:  # Top 10 sets
            click.echo(f"   • {set_code or 'Unknown'}: {count}")
        
        if len(set_items) > 10:
            click.echo(f"   • ... and {len(set_items) - 10} more sets")
    
    except Exception as e:
        click.echo(f"💥 Stats error: {str(e)}")

@database.command('search')
@click.argument('search_term')
@click.option('--collection', '-c', help='Search within specific collection')
@click.option('--set-code', '-s', help='Filter by set code')
@click.option('--rarity', '-r', help='Filter by rarity')
@click.option('--colors', help='Filter by colors (e.g., "WU" for white/blue)')
@click.option('--limit', '-l', type=int, default=10, help='Maximum results to show')
def search_cards(search_term, collection, set_code, rarity, colors, limit):
    """Search for cards in the database"""
    
    click.echo(f"🔍 Searching for: {search_term}")
    
    try:
        processor = MTGCardProcessingSystem()
        
        filters = {}
        if collection:
            filters['collection'] = collection
        if set_code:
            filters['set_code'] = set_code
        if rarity:
            filters['rarity'] = rarity
        if colors:
            filters['colors'] = colors
        
        results = processor.search_cards(search_term, filters)
        
        if not results:
            click.echo("❌ No cards found matching your search")
            return
        
        click.echo(f"🎯 Found {len(results)} results (showing first {min(limit, len(results))}):")
        
        for i, card in enumerate(results[:limit], 1):
            name = card['name']
            mana_cost = card['mana_cost'] or ''
            set_code = card['set_code'] or 'UNK'
            rarity = card['rarity'] or 'unknown'
            collection = card['collection_name'] or 'unknown'
            
            click.echo(f"{i:2d}. {name} {mana_cost} ({set_code}) - {rarity}")
            click.echo(f"     Collection: {collection}")
            
            if card['oracle_text']:
                text = card['oracle_text'][:80] + "..." if len(card['oracle_text']) > 80 else card['oracle_text']
                click.echo(f"     {text}")
            click.echo()
    
    except Exception as e:
        click.echo(f"💥 Search error: {str(e)}")

@cli.command()
@click.option('--days', '-d', type=int, default=30, help='Delete files older than N days')
@click.option('--dry-run', is_flag=True, help='Show what would be deleted without deleting')
def cleanup(days, dry_run):
    """Clean up old processing files"""
    
    if dry_run:
        click.echo(f"🔍 Dry run: showing files that would be deleted (older than {days} days)")
    else:
        click.echo(f"🧹 Cleaning up files older than {days} days...")
    
    try:
        if not dry_run:
            cleaned_count = cleanup_old_files(days)
            click.echo(f"✅ Cleaned up {cleaned_count} files")
        else:
            # Implement dry run logic
            click.echo("📋 Files that would be deleted:")
            click.echo("   (Dry run functionality not yet implemented)")
    
    except Exception as e:
        click.echo(f"💥 Cleanup error: {str(e)}")

@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
def identify(image_path):
    """Identify a single card image"""
    
    click.echo(f"🎯 Identifying card: {image_path}")
    
    try:
        processor = MTGCardProcessingSystem()
        result = processor.process_single_card_image(image_path)
        
        if result.get('success'):
            extracted = result['extracted_data']
            identified = result['identification_result']
            
            click.echo("✅ Card identified!")
            click.echo(f"\n📋 Extracted Information:")
            click.echo(f"   • Name: {extracted.get('name', 'Unknown')}")
            click.echo(f"   • Mana Cost: {extracted.get('mana_cost', 'Unknown')}")
            click.echo(f"   • Type: {extracted.get('type_line', 'Unknown')}")
            
            if extracted.get('power_toughness'):
                pt = extracted['power_toughness']
                click.echo(f"   • Power/Toughness: {pt.get('power', '?')}/{pt.get('toughness', '?')}")
            
            click.echo(f"   • Confidence: {extracted.get('confidence', 0):.2f}")
            
            if identified.get('success'):
                card_data = identified['card_data']
                click.echo(f"\n🎲 API Identification:")
                click.echo(f"   • Confirmed Name: {card_data.get('name', 'Unknown')}")
                click.echo(f"   • Set: {card_data.get('set_name', 'Unknown')} ({card_data.get('set', 'UNK')})")
                click.echo(f"   • Rarity: {card_data.get('rarity', 'Unknown')}")
                click.echo(f"   • Method: {identified.get('method', 'Unknown')}")
                click.echo(f"   • Confidence: {identified.get('confidence', 0):.2f}")
        else:
            click.echo(f"❌ Identification failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        click.echo(f"💥 Identification error: {str(e)}")

@cli.command()
def version():
    """Show version information"""
    click.echo("🎴 MTG Card Processing System")
    click.echo("Version: 1.0.0")
    click.echo("Author: AI Assistant")
    click.echo("\nComponents:")
    click.echo("• PDF Processing: PyMuPDF")
    click.echo("• OCR: EasyOCR/Tesseract")
    click.echo("• Computer Vision: OpenCV")
    click.echo("• API: Scryfall")
    click.echo("• Database: PostgreSQL/SQLAlchemy")

@cli.group()
def debug():
    """Debug and diagnostic commands"""
    pass

@debug.command('pdf')
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--page', type=int, default=None, help='Debug specific page (0-indexed)')
@click.option('--save-images', is_flag=True, default=True, help='Save debug images')
def debug_pdf(pdf_path, page, save_images):
    """Debug PDF processing step by step"""
    from debug_tool import DebugTool
    
    debug_tool = DebugTool()
    debug_tool.debug_pdf(pdf_path, save_images)

@debug.command('image')
@click.argument('image_path', type=click.Path(exists=True))
def debug_image(image_path):
    """Debug single card image OCR"""
    from debug_tool import DebugTool
    
    debug_tool = DebugTool()
    debug_tool.debug_single_image(image_path)

@debug.command('ocr')
@click.argument('image_path', type=click.Path(exists=True))
def debug_ocr(image_path):
    """Test different OCR engines on image"""
    from debug_tool import DebugTool
    
    debug_tool = DebugTool()
    debug_tool.test_ocr_engines(image_path)

@debug.command('visualize')
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--page', type=int, default=0, help='Page number to visualize')
def debug_visualize(pdf_path, page):
    """Create visual debugging output"""
    from debug_tool import DebugTool
    
    debug_tool = DebugTool()
    debug_tool.visualize_detection(pdf_path, page)

@cli.command()
def config():
    """Show current configuration"""
    click.echo("⚙️  Current Configuration:")
    click.echo(f"\n📁 Paths:")
    click.echo(f"   • Storage: {settings.STORAGE_BASE_PATH}")
    click.echo(f"   • Card Images: {settings.CARD_IMAGES_PATH}")
    click.echo(f"   • Exports: {settings.EXPORTS_PATH}")
    
    click.echo(f"\n🔧 Processing:")
    click.echo(f"   • OCR Engine: {settings.OCR_ENGINE}")
    click.echo(f"   • Expected Grid: {settings.EXPECTED_GRID_ROWS}x{settings.EXPECTED_GRID_COLS}")
    click.echo(f"   • Batch Size: {settings.BATCH_SIZE}")
    
    click.echo(f"\n🎯 Detection:")
    click.echo(f"   • Duplicate Threshold: {settings.DUPLICATE_SIMILARITY_THRESHOLD}")
    click.echo(f"   • OCR Confidence: {settings.OCR_CONFIDENCE_THRESHOLD}")
    
    click.echo(f"\n🗄️  Database:")
    click.echo(f"   • URL: {settings.DATABASE_URL}")
    click.echo(f"   • Pool Size: {settings.DATABASE_POOL_SIZE}")

if __name__ == '__main__':
    cli()