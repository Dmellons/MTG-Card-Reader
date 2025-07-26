"""
Export Manager Module
Handles exporting card data to various formats (CSV, JSON, database dumps)
"""
import os
import json
import csv
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session
from models import DatabaseManager, CardRepository, Card
from config import settings

logger = logging.getLogger(__name__)

class ExportManager:
    """Manages export of card data to various formats"""
    
    def __init__(self):
        self.db_manager = DatabaseManager(settings.DATABASE_URL)
        self.exports_path = settings.EXPORTS_PATH
        
        # Ensure exports directory exists
        os.makedirs(self.exports_path, exist_ok=True)
    
    def export_collection_to_csv(self, collection_name: str = None, 
                                output_file: str = None) -> Dict:
        """Export cards to CSV format"""
        session = self.db_manager.get_session()
        card_repo = CardRepository(session)
        
        try:
            # Generate filename if not provided
            if not output_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                collection_part = f"{collection_name}_" if collection_name else "all_"
                output_file = os.path.join(self.exports_path, f"{collection_part}{timestamp}.csv")
            
            # Get cards data
            if collection_name:
                cards = card_repo.get_cards_by_collection(collection_name)
            else:
                cards = session.query(Card).all()
            
            if not cards:
                return {
                    'success': False,
                    'error': 'No cards found for export',
                    'collection_name': collection_name
                }
            
            # Convert to list of dictionaries
            cards_data = []
            for card in cards:
                cards_data.append(self._card_to_export_dict(card))
            
            # Create DataFrame and export
            df = pd.DataFrame(cards_data)
            
            # Reorder columns for better readability
            column_order = [
                'name', 'mana_cost', 'converted_mana_cost', 'type_line', 
                'oracle_text', 'flavor_text', 'power', 'toughness',
                'set_code', 'set_name', 'rarity', 'collector_number',
                'colors', 'color_identity', 'collection_name',
                'confidence_score', 'identification_method', 'processed_date'
            ]
            
            # Only include columns that exist
            available_columns = [col for col in column_order if col in df.columns]
            df = df[available_columns]
            
            # Export to CSV
            df.to_csv(output_file, index=False, encoding='utf-8')
            
            logger.info(f"Exported {len(cards_data)} cards to CSV: {output_file}")
            
            return {
                'success': True,
                'file_path': output_file,
                'cards_exported': len(cards_data),
                'collection_name': collection_name,
                'export_format': 'csv'
            }
            
        except Exception as e:
            logger.error(f"CSV export failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'collection_name': collection_name
            }
        finally:
            session.close()
    
    def export_collection_to_json(self, collection_name: str = None,
                                 output_file: str = None, 
                                 include_metadata: bool = True) -> Dict:
        """Export cards to JSON format"""
        session = self.db_manager.get_session()
        card_repo = CardRepository(session)
        
        try:
            # Generate filename if not provided
            if not output_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                collection_part = f"{collection_name}_" if collection_name else "all_"
                output_file = os.path.join(self.exports_path, f"{collection_part}{timestamp}.json")
            
            # Get cards data
            if collection_name:
                cards = card_repo.get_cards_by_collection(collection_name)
            else:
                cards = session.query(Card).all()
            
            if not cards:
                return {
                    'success': False,
                    'error': 'No cards found for export',
                    'collection_name': collection_name
                }
            
            # Convert to export format
            cards_data = [self._card_to_export_dict(card) for card in cards]
            
            # Create export structure
            export_data = {
                'cards': cards_data
            }
            
            if include_metadata:
                stats = card_repo.get_collection_stats(collection_name)
                export_data['metadata'] = {
                    'collection_name': collection_name,
                    'export_date': datetime.now().isoformat(),
                    'total_cards': len(cards_data),
                    'statistics': stats,
                    'export_format': 'json',
                    'schema_version': '1.0'
                }
            
            # Export to JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Exported {len(cards_data)} cards to JSON: {output_file}")
            
            return {
                'success': True,
                'file_path': output_file,
                'cards_exported': len(cards_data),
                'collection_name': collection_name,
                'export_format': 'json'
            }
            
        except Exception as e:
            logger.error(f"JSON export failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'collection_name': collection_name
            }
        finally:
            session.close()
    
    def generate_collection_report(self, collection_name: str = None,
                                 output_file: str = None) -> Dict:
        """Generate comprehensive collection report"""
        session = self.db_manager.get_session()
        card_repo = CardRepository(session)
        
        try:
            # Generate filename if not provided
            if not output_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                collection_part = f"{collection_name}_" if collection_name else "all_"
                output_file = os.path.join(self.exports_path, f"{collection_part}report_{timestamp}.json")
            
            # Get comprehensive statistics
            stats = card_repo.get_collection_stats(collection_name)
            
            # Get additional analytics
            analytics = self._generate_analytics(collection_name, session)
            
            # Create report structure
            report = {
                'report_metadata': {
                    'collection_name': collection_name or 'All Collections',
                    'generated_date': datetime.now().isoformat(),
                    'report_type': 'collection_analysis'
                },
                'summary': {
                    'total_cards': stats['total_cards'],
                    'unique_cards': stats['unique_cards'],
                    'duplicate_ratio': ((stats['total_cards'] - stats['unique_cards']) / 
                                       stats['total_cards'] * 100) if stats['total_cards'] > 0 else 0
                },
                'distribution_analysis': {
                    'rarity_distribution': stats['rarity_distribution'],
                    'set_distribution': stats['set_distribution'],
                    'color_distribution': analytics['color_distribution'],
                    'type_distribution': analytics['type_distribution']
                },
                'quality_metrics': {
                    'confidence_analysis': analytics['confidence_analysis'],
                    'identification_methods': analytics['identification_methods'],
                    'processing_quality': analytics['processing_quality']
                },
                'recommendations': self._generate_recommendations(stats, analytics)
            }
            
            # Save report
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Generated collection report: {output_file}")
            
            return {
                'success': True,
                'file_path': output_file,
                'collection_name': collection_name,
                'report_summary': report['summary']
            }
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'collection_name': collection_name
            }
        finally:
            session.close()
    
    def export_to_database_dump(self, collection_name: str = None,
                               format: str = 'sql') -> Dict:
        """Export collection to database dump format"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            collection_part = f"{collection_name}_" if collection_name else "all_"
            
            if format == 'sql':
                output_file = os.path.join(self.exports_path, f"{collection_part}dump_{timestamp}.sql")
                return self._export_sql_dump(collection_name, output_file)
            elif format == 'sqlite':
                output_file = os.path.join(self.exports_path, f"{collection_part}backup_{timestamp}.db")
                return self._export_sqlite_backup(collection_name, output_file)
            else:
                return {
                    'success': False,
                    'error': f'Unsupported export format: {format}'
                }
                
        except Exception as e:
            logger.error(f"Database dump failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _card_to_export_dict(self, card: Card) -> Dict:
        """Convert Card model to export dictionary"""
        return {
            'id': str(card.id),
            'name': card.name,
            'scryfall_id': str(card.scryfall_id) if card.scryfall_id else None,
            'mana_cost': card.mana_cost,
            'converted_mana_cost': card.converted_mana_cost,
            'type_line': card.type_line,
            'oracle_text': card.oracle_text,
            'flavor_text': card.flavor_text,
            'power': card.power,
            'toughness': card.toughness,
            'set_code': card.set_code,
            'set_name': card.set_name,
            'rarity': card.rarity,
            'collector_number': card.collector_number,
            'colors': card.colors,
            'color_identity': card.color_identity,
            'collection_name': card.collection_name,
            'image_path': card.image_path,
            'source_pdf': card.source_pdf,
            'source_page': card.source_page,
            'card_position': card.card_position,
            'processed_date': card.processed_date.isoformat() if card.processed_date else None,
            'confidence_score': card.confidence_score,
            'identification_method': card.identification_method
        }
    
    def _generate_analytics(self, collection_name: str, session: Session) -> Dict:
        """Generate detailed analytics for the collection"""
        from sqlalchemy import func
        
        query = session.query(Card)
        if collection_name:
            query = query.filter(Card.collection_name == collection_name)
        
        # Color distribution
        color_counts = {}
        cards_with_colors = query.filter(Card.colors.isnot(None)).all()
        for card in cards_with_colors:
            colors = card.colors or ''
            for color in colors:
                color_counts[color] = color_counts.get(color, 0) + 1
        
        # Type distribution
        type_counts = {}
        cards_with_types = query.filter(Card.type_line.isnot(None)).all()
        for card in cards_with_types:
            type_line = card.type_line or ''
            # Extract main types
            main_types = ['Creature', 'Instant', 'Sorcery', 'Enchantment', 'Artifact', 'Planeswalker', 'Land']
            for card_type in main_types:
                if card_type.lower() in type_line.lower():
                    type_counts[card_type] = type_counts.get(card_type, 0) + 1
        
        # Confidence analysis
        confidence_stats = query.with_entities(
            func.avg(Card.confidence_score).label('avg_confidence'),
            func.min(Card.confidence_score).label('min_confidence'),
            func.max(Card.confidence_score).label('max_confidence')
        ).first()
        
        # Identification methods breakdown
        method_counts = dict(query.with_entities(
            Card.identification_method,
            func.count(Card.id)
        ).group_by(Card.identification_method).all())
        
        # Processing quality metrics
        total_cards = query.count()
        high_confidence_cards = query.filter(Card.confidence_score >= 0.8).count()
        low_confidence_cards = query.filter(Card.confidence_score < 0.5).count()
        
        return {
            'color_distribution': color_counts,
            'type_distribution': type_counts,
            'confidence_analysis': {
                'average_confidence': float(confidence_stats.avg_confidence or 0),
                'min_confidence': float(confidence_stats.min_confidence or 0),
                'max_confidence': float(confidence_stats.max_confidence or 0),
                'high_confidence_percentage': (high_confidence_cards / total_cards * 100) if total_cards > 0 else 0,
                'low_confidence_percentage': (low_confidence_cards / total_cards * 100) if total_cards > 0 else 0
            },
            'identification_methods': method_counts,
            'processing_quality': {
                'total_processed': total_cards,
                'high_confidence_count': high_confidence_cards,
                'low_confidence_count': low_confidence_cards
            }
        }
    
    def _generate_recommendations(self, stats: Dict, analytics: Dict) -> List[str]:
        """Generate recommendations based on collection analysis"""
        recommendations = []
        
        # Check confidence levels
        avg_confidence = analytics['confidence_analysis']['average_confidence']
        if avg_confidence < 0.6:
            recommendations.append(
                "Low average confidence detected. Consider improving image quality or OCR settings."
            )
        
        # Check for missing data
        total_cards = stats['total_cards']
        if total_cards > 0:
            cards_without_sets = sum(1 for set_code, count in stats['set_distribution'].items() 
                                   if set_code in ['UNK', 'unknown', None])
            if cards_without_sets / total_cards > 0.2:
                recommendations.append(
                    "High percentage of cards without set information. Consider manual review."
                )
        
        # Check identification method distribution
        method_dist = analytics['identification_methods']
        ocr_only_ratio = method_dist.get('ocr_only', 0) / total_cards if total_cards > 0 else 0
        if ocr_only_ratio > 0.5:
            recommendations.append(
                "Many cards identified by OCR only. Consider building image hash database for better accuracy."
            )
        
        # Check for processing quality
        low_conf_percentage = analytics['confidence_analysis']['low_confidence_percentage']
        if low_conf_percentage > 15:
            recommendations.append(
                f"{low_conf_percentage:.1f}% of cards have low confidence. Manual review recommended."
            )
        
        if not recommendations:
            recommendations.append("Collection processing appears to be of good quality!")
        
        return recommendations
    
    def _export_sql_dump(self, collection_name: str, output_file: str) -> Dict:
        """Export collection as SQL dump"""
        session = self.db_manager.get_session()
        
        try:
            query = session.query(Card)
            if collection_name:
                query = query.filter(Card.collection_name == collection_name)
            
            cards = query.all()
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("-- MTG Card Collection SQL Dump\n")
                f.write(f"-- Generated: {datetime.now().isoformat()}\n")
                f.write(f"-- Collection: {collection_name or 'All'}\n\n")
                
                for card in cards:
                    insert_sql = self._generate_insert_sql(card)
                    f.write(insert_sql + "\n")
            
            return {
                'success': True,
                'file_path': output_file,
                'cards_exported': len(cards),
                'export_format': 'sql'
            }
            
        except Exception as e:
            logger.error(f"SQL dump failed: {str(e)}")
            return {'success': False, 'error': str(e)}
        finally:
            session.close()
    
    def _export_sqlite_backup(self, collection_name: str, output_file: str) -> Dict:
        """Export collection to SQLite database"""
        import sqlite3
        
        session = self.db_manager.get_session()
        
        try:
            # Create SQLite database
            sqlite_conn = sqlite3.connect(output_file)
            sqlite_cursor = sqlite_conn.cursor()
            
            # Create table
            create_table_sql = """
            CREATE TABLE cards (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                mana_cost TEXT,
                converted_mana_cost INTEGER,
                type_line TEXT,
                oracle_text TEXT,
                flavor_text TEXT,
                power TEXT,
                toughness TEXT,
                set_code TEXT,
                set_name TEXT,
                rarity TEXT,
                colors TEXT,
                collection_name TEXT,
                confidence_score REAL,
                processed_date TEXT
            )
            """
            sqlite_cursor.execute(create_table_sql)
            
            # Insert data
            query = session.query(Card)
            if collection_name:
                query = query.filter(Card.collection_name == collection_name)
            
            cards = query.all()
            
            for card in cards:
                card_data = (
                    str(card.id), card.name, card.mana_cost, card.converted_mana_cost,
                    card.type_line, card.oracle_text, card.flavor_text,
                    card.power, card.toughness, card.set_code, card.set_name,
                    card.rarity, card.colors, card.collection_name,
                    card.confidence_score, 
                    card.processed_date.isoformat() if card.processed_date else None
                )
                
                sqlite_cursor.execute("""
                    INSERT INTO cards VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, card_data)
            
            sqlite_conn.commit()
            sqlite_conn.close()
            
            return {
                'success': True,
                'file_path': output_file,
                'cards_exported': len(cards),
                'export_format': 'sqlite'
            }
            
        except Exception as e:
            logger.error(f"SQLite backup failed: {str(e)}")
            return {'success': False, 'error': str(e)}
        finally:
            session.close()
    
    def _generate_insert_sql(self, card: Card) -> str:
        """Generate SQL INSERT statement for a card"""
        def sql_escape(value):
            if value is None:
                return 'NULL'
            if isinstance(value, str):
                return f"'{value.replace("\"", "\'\"")}'"
            return str(value)
        
        values = [
            sql_escape(str(card.id)),
            sql_escape(card.name),
            sql_escape(card.mana_cost),
            sql_escape(card.converted_mana_cost),
            sql_escape(card.type_line),
            sql_escape(card.oracle_text),
            sql_escape(card.flavor_text),
            sql_escape(card.power),
            sql_escape(card.toughness),
            sql_escape(card.set_code),
            sql_escape(card.set_name),
            sql_escape(card.rarity),
            sql_escape(card.colors),
            sql_escape(card.collection_name),
            sql_escape(card.confidence_score),
            sql_escape(card.processed_date.isoformat() if card.processed_date else None)
        ]
        
        return f"INSERT INTO cards VALUES ({', '.join(values)});"

class BulkExporter:
    """Handles bulk export operations across multiple collections"""
    
    def __init__(self):
        self.export_manager = ExportManager()
    
    def export_all_collections(self, formats: List[str] = ['csv', 'json']) -> Dict:
        """Export all collections in specified formats"""
        session = self.export_manager.db_manager.get_session()
        
        try:
            # Get all unique collection names
            collections = session.query(Card.collection_name).distinct().all()
            collection_names = [c[0] for c in collections if c[0]]
            
            results = {}
            
            for collection_name in collection_names:
                results[collection_name] = {}
                
                for format_type in formats:
                    if format_type == 'csv':
                        result = self.export_manager.export_collection_to_csv(collection_name)
                    elif format_type == 'json':
                        result = self.export_manager.export_collection_to_json(collection_name)
                    elif format_type == 'report':
                        result = self.export_manager.generate_collection_report(collection_name)
                    else:
                        result = {'success': False, 'error': f'Unknown format: {format_type}'}
                    
                    results[collection_name][format_type] = result
            
            return {
                'success': True,
                'collections_processed': len(collection_names),
                'formats': formats,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Bulk export failed: {str(e)}")
            return {'success': False, 'error': str(e)}
        finally:
            session.close()