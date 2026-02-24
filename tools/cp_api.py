# ==========================================================================
# Master Thesis
#   - André Filipe Gomes Silvestre, 20240502
#
# CP (Comboios de Portugal) Train API Functions
# This module provides:
#   - Static GTFS data from CP's official feed (schedules, routes, stops)
#   - Real-time train status from comboios.live API
#   - Integration of static + real-time data for comprehensive train info
#
# GTFS Source: https://publico.cp.pt/gtfs/gtfs.zip
# Real-time API: https://comboios.live/api/
# Lines covered: Cascais, Sintra, Azambuja, Fertagus, Sado
# ==========================================================================

# Required libraries:
# pip install requests langchain-core

import csv
import io
import json
import logging
import os
import sqlite3
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from langchain_core.tools import tool

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================================================
# Constants and Configuration
# ==========================================================================

# CP GTFS Static Data
CP_GTFS_URL = "https://publico.cp.pt/gtfs/gtfs.zip"

# Real-time API endpoints (comboios.live)
CP_STATIONS_URL = "https://comboios.live/api/stations"
CP_VEHICLES_URL = "https://comboios.live/api/vehicles"

# Data storage paths
DATA_DIR = Path(__file__).parent.parent / "data" / "cp"
DB_PATH = DATA_DIR / "cp_gtfs.db"
METADATA_PATH = DATA_DIR / "metadata.json"
GTFS_ZIP_PATH = DATA_DIR / "gtfs.zip"

# Cache settings
CACHE_EXPIRATION_HOURS = 1  # Real-time cache
GTFS_REFRESH_DAYS = 1  # Check for GTFS updates daily
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
BACKOFF_FACTOR = 2

# In-memory caches
_cp_stations_cache: Dict[str, Dict[str, Any]] = {}
_cp_stations_last_load: Optional[datetime] = None

# Lisbon Metropolitan Area (AML) bounding box
# Covers: Cascais, Sintra, Mafra, Loures, Odivelas, Amadora, Oeiras, Lisboa,
#         Almada, Seixal, Barreiro, Moita, Montijo, Alcochete, Setúbal, Palmela, Sesimbra
AML_BOUNDS = {
    'lat_min': 38.4,  # Southern limit (Setúbal area)
    'lat_max': 39.0,  # Northern limit (Mafra area)
    'lon_min': -9.5,  # Western limit (Cascais area)
    'lon_max': -8.7   # Eastern limit (Montijo area)
}

# Check area coverage: https://www.keene.edu/campus/maps/tool/?coordinates=-9.5000000%2C%2038.4000000%0A-8.7000000%2C%2038.4000000%0A-8.7000000%2C%2039.0000000%0A-9.5000000%2C%2039.0000000%0A-9.5000000%2C%2038.4000000

# CP Lines serving the AML region
CP_LINES = {
    "cascais": {
        "name": "Linha de Cascais",
        "terminal_a": "Cais do Sodré",
        "terminal_b": "Cascais",
        "description": "Coastal line serving Lisbon's western suburbs and beaches"
    },
    "sintra": {
        "name": "Linha de Sintra",
        "terminal_a": "Rossio / Oriente",
        "terminal_b": "Sintra",
        "description": "Serves the historic town of Sintra and northwestern suburbs"
    },
    "azambuja": {
        "name": "Linha de Azambuja",
        "terminal_a": "Santa Apolónia / Oriente",
        "terminal_b": "Azambuja",
        "description": "Serves northeastern suburbs and connects to Porto line"
    },
    "norte": {
        "name": "Linha do Norte",
        "terminal_a": "Lisboa Santa Apolónia / Oriente",
        "terminal_b": "Porto Campanhã",
        "description": "Main long-distance line connecting Lisbon to Porto"
    },
    "fertagus": {
        "name": "Fertagus",
        "terminal_a": "Roma-Areeiro",
        "terminal_b": "Setúbal",
        "description": "Crosses Tagus river via 25 de Abril Bridge"
    },
    "sado": {
        "name": "Linha do Sado",
        "terminal_a": "Barreiro",
        "terminal_b": "Setúbal",
        "description": "Connects Barreiro ferry terminal to Setúbal"
    }
}

# Key CP stations in the AML (for quick reference)
CP_KEY_STATIONS = {
    # Main hubs
    "oriente": {"name": "Lisboa - Oriente", "lines": ["sintra", "azambuja", "norte"], "metro": "vermelha"},
    "entrecampos": {"name": "Entrecampos", "lines": ["fertagus", "norte"], "metro": "amarela"},
    "rossio": {"name": "Rossio", "lines": ["sintra"], "metro": "verde"},
    "cais_sodre": {"name": "Cais do Sodré", "lines": ["cascais"], "metro": "verde"},
    "santa_apolonia": {"name": "Santa Apolónia", "lines": ["azambuja", "norte"], "metro": "azul"},
    
    # Cascais Line
    "cascais": {"name": "Cascais", "lines": ["cascais"], "description": "Western terminus, beach town"},
    "estoril": {"name": "Estoril", "lines": ["cascais"], "description": "Casino and beach resort"},
    "oeiras": {"name": "Oeiras", "lines": ["cascais"], "description": "Business district"},
    "belem": {"name": "Belém", "lines": ["cascais"], "description": "Near UNESCO monuments"},
    
    # Sintra Line
    "sintra": {"name": "Sintra", "lines": ["sintra"], "description": "UNESCO World Heritage site"},
    "queluz": {"name": "Queluz-Belas", "lines": ["sintra"], "description": "Near Queluz Palace"},
    "amadora": {"name": "Amadora", "lines": ["sintra"], "description": "Suburban hub"},
    
    # Azambuja Line
    "vila_franca": {"name": "Vila Franca de Xira", "lines": ["azambuja"], "description": "Northern suburbs"},
    "alverca": {"name": "Alverca", "lines": ["azambuja"], "description": "Industrial zone"},
    
    # South Bank
    "barreiro": {"name": "Barreiro", "lines": ["sado"], "description": "Ferry connection to Lisboa"},
    "setubal": {"name": "Setúbal", "lines": ["sado", "fertagus"], "description": "Southern city"},
    "pragal": {"name": "Pragal", "lines": ["fertagus"], "description": "South bank, near Almada"},
}

# Alias for backward compatibility with transport_api.py
CP_STATIONS = CP_KEY_STATIONS


def get_cp_station_info(station_name: str) -> Optional[Dict[str, Any]]:
    """
    Returns information about a CP train station from the key stations list.
    
    Args:
        station_name: Name of the station (case-insensitive).
        
    Returns:
        Station information or None if not found.
    """
    station_lower = station_name.lower().strip()
    
    # Try direct match first
    if station_lower in CP_KEY_STATIONS:
        return CP_KEY_STATIONS[station_lower]
    
    # Try partial match
    for key, info in CP_KEY_STATIONS.items():
        if station_lower in info['name'].lower() or info['name'].lower() in station_lower:
            return info
        # Also check without accents
        if station_lower.replace('ó', 'o').replace('ã', 'a') in key:
            return info
    
    return None


# ==========================================================================
# CP GTFS Manager Class
# ==========================================================================

class CPGTFSManager:
    """
    Manages CP GTFS static data with SQLite storage.
    
    Features:
    - Downloads GTFS feed from CP's official URL
    - Stores data in SQLite for efficient querying
    - Implements time-based refresh (daily check)
    - Smart update detection using HTTP ETag/Last-Modified headers
    - Provides query methods for schedules, routes, and stops
    """
    
    def __init__(self, data_dir: Path = DATA_DIR):
        """
        Initializes the GTFS manager.
        
        Args:
            data_dir: Directory for storing GTFS data and SQLite DB.
        """
        self.data_dir = data_dir
        self.db_path = data_dir / "cp_gtfs.db"
        self.metadata_path = data_dir / "metadata.json"
        self.gtfs_zip_path = data_dir / "gtfs.zip"
        
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Loads metadata from JSON file."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
    
    def _save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Saves metadata to JSON file."""
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def check_for_updates(self) -> bool:
        """
        Checks if GTFS data needs to be refreshed using HTTP headers.
        
        Uses HTTP HEAD request to check Last-Modified or ETag headers
        from the server, avoiding unnecessary downloads when data hasn't changed.
        Falls back to time-based refresh if headers are unavailable.
        
        Returns:
            True if update is needed, False otherwise.
        """
        metadata = self._load_metadata()
        last_download = metadata.get('last_download')
        cached_etag = metadata.get('etag')
        cached_last_modified = metadata.get('last_modified')
        
        # If no previous download, update is needed
        if not last_download:
            logger.info("No previous GTFS download found. Update needed.")
            return True
        
        # Try HTTP HEAD request to check server's Last-Modified/ETag
        try:
            logger.info("Checking GTFS server for updates (HEAD request)...")
            head_response = requests.head(CP_GTFS_URL, timeout=10, allow_redirects=True)
            head_response.raise_for_status()
            
            server_etag = head_response.headers.get('ETag')
            server_last_modified = head_response.headers.get('Last-Modified')
            
            # Check ETag first (most reliable)
            if server_etag and cached_etag:
                if server_etag == cached_etag:
                    logger.info(f"\033[1;32m✅ ETag unchanged ({server_etag[:20]}...). No update needed.\033[0m")
                    return False
                else:
                    logger.info(f"ETag changed: {cached_etag[:20]}... → {server_etag[:20]}... Update needed.")
                    return True
            
            # Check Last-Modified header
            if server_last_modified and cached_last_modified:
                if server_last_modified == cached_last_modified:
                    logger.info(f"\033[1;32m✅ Last-Modified unchanged ({server_last_modified}). No update needed.\033[0m")
                    return False
                else:
                    logger.info(f"Last-Modified changed: {cached_last_modified} → {server_last_modified}. Update needed.")
                    return True
            
            # If server provides new headers we didn't have, log and update
            if server_etag or server_last_modified:
                logger.info(f"Server headers available (ETag: {bool(server_etag)}, Last-Modified: {bool(server_last_modified)})")
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"HEAD request failed: {e}. Falling back to time-based check.")
        
        # Fallback: time-based refresh
        try:
            last_download_dt = datetime.fromisoformat(last_download)
            days_since = (datetime.now() - last_download_dt).days
            
            if days_since >= GTFS_REFRESH_DAYS:
                logger.info(f"GTFS data is {days_since} days old. Update needed (time-based fallback).")
                return True
            
            logger.info(f"GTFS data is {days_since} days old. No update needed (time-based fallback).")
            return False
            
        except (ValueError, TypeError):
            logger.warning("Invalid last_download timestamp. Update needed.")
            return True
    
    def download_gtfs(self) -> bool:
        """
        Downloads CP GTFS feed from the official URL.
        
        Captures HTTP headers (Last-Modified, ETag) for smart update checking.
        
        Returns:
            True if download successful, False otherwise.
        """
        logger.info(f"Downloading CP GTFS from {CP_GTFS_URL}...")
        
        try:
            response = requests.get(CP_GTFS_URL, timeout=60, stream=True)
            response.raise_for_status()
            
            # Capture server headers for future update checks
            server_etag = response.headers.get('ETag')
            server_last_modified = response.headers.get('Last-Modified')
            content_length = response.headers.get('Content-Length', 'unknown')
            
            logger.info(f"  Content-Length: {content_length} bytes")
            if server_last_modified:
                logger.info(f"  Last-Modified: {server_last_modified}")
            if server_etag:
                logger.info(f"  ETag: {server_etag[:30]}...")
            
            # Save to file
            with open(self.gtfs_zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Verify it's a valid ZIP
            if not zipfile.is_zipfile(self.gtfs_zip_path):
                logger.error("Downloaded file is not a valid ZIP")
                return False
            
            # Store server headers in metadata for future update checks
            metadata = self._load_metadata()
            metadata['last_download'] = datetime.now().isoformat()
            if server_etag:
                metadata['etag'] = server_etag
            if server_last_modified:
                metadata['last_modified'] = server_last_modified
            metadata['content_length'] = content_length
            self._save_metadata(metadata)
            
            logger.info("\033[1;32m✅ GTFS downloaded successfully\033[0m")
            return True
            
        except requests.exceptions.Timeout:
            logger.error("Timeout downloading CP GTFS (60s)")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading CP GTFS: {e}")
            return False
        except IOError as e:
            logger.error(f"Error saving GTFS file: {e}")
            return False
    
    def convert_to_sqlite(self) -> bool:
        """
        Converts GTFS ZIP to SQLite database with optimized schema.
        
        Creates tables: agency, calendar, calendar_dates, routes, stops,
                       trips, stop_times, shapes (if available)
        
        Returns:
            True if conversion successful, False otherwise.
        """
        if not self.gtfs_zip_path.exists():
            logger.error("GTFS ZIP file not found")
            return False
        
        logger.info("Converting GTFS to SQLite...")
        
        # Remove old database if exists
        if self.db_path.exists():
            self.db_path.unlink()
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Enable optimizations
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            
            # Create tables with proper schema
            self._create_tables(cursor)
            
            # Read and insert GTFS data
            with zipfile.ZipFile(self.gtfs_zip_path, 'r') as zf:
                gtfs_files = {
                    'agency.txt': self._insert_agency,
                    'calendar.txt': self._insert_calendar,
                    'calendar_dates.txt': self._insert_calendar_dates,
                    'routes.txt': self._insert_routes,
                    'stops.txt': self._insert_stops,
                    'trips.txt': self._insert_trips,
                    'stop_times.txt': self._insert_stop_times,
                    'shapes.txt': self._insert_shapes,
                }
                
                for filename, insert_func in gtfs_files.items():
                    if filename in zf.namelist():
                        logger.info(f"  Processing {filename}...")
                        try:
                            with zf.open(filename) as f:
                                reader = csv.DictReader(
                                    io.TextIOWrapper(f, encoding='utf-8-sig')
                                )
                                insert_func(cursor, reader)
                        except Exception as e:
                            logger.warning(f"  Error processing {filename}: {e}")
                    else:
                        logger.warning(f"  {filename} not found in GTFS")
            
            # Create indexes for fast queries
            self._create_indexes(cursor)
            
            conn.commit()
            conn.close()
            
            # Update metadata with DB creation time (preserve download headers)
            metadata = self._load_metadata()
            metadata['db_created'] = datetime.now().isoformat()
            self._save_metadata(metadata)
            
            logger.info(f"\033[1;32m✅ SQLite database created at {self.db_path}\033[0m")
            return True
            
        except Exception as e:
            logger.error(f"Error converting GTFS to SQLite: {e}")
            return False
    
    def _create_tables(self, cursor: sqlite3.Cursor) -> None:
        """Creates GTFS tables with proper schema."""
        
        # Agency table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agency (
                agency_id TEXT PRIMARY KEY,
                agency_name TEXT NOT NULL,
                agency_url TEXT,
                agency_timezone TEXT,
                agency_lang TEXT,
                agency_phone TEXT
            )
        """)
        
        # Calendar table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calendar (
                service_id TEXT PRIMARY KEY,
                monday INTEGER,
                tuesday INTEGER,
                wednesday INTEGER,
                thursday INTEGER,
                friday INTEGER,
                saturday INTEGER,
                sunday INTEGER,
                start_date TEXT,
                end_date TEXT
            )
        """)
        
        # Calendar dates table (exceptions)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calendar_dates (
                service_id TEXT,
                date TEXT,
                exception_type INTEGER,
                PRIMARY KEY (service_id, date)
            )
        """)
        
        # Routes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS routes (
                route_id TEXT PRIMARY KEY,
                agency_id TEXT,
                route_short_name TEXT,
                route_long_name TEXT,
                route_desc TEXT,
                route_type INTEGER,
                route_color TEXT,
                route_text_color TEXT
            )
        """)
        
        # Stops table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stops (
                stop_id TEXT PRIMARY KEY,
                stop_code TEXT,
                stop_name TEXT NOT NULL,
                stop_desc TEXT,
                stop_lat REAL,
                stop_lon REAL,
                zone_id TEXT,
                stop_url TEXT,
                location_type INTEGER,
                parent_station TEXT,
                platform_code TEXT
            )
        """)
        
        # Trips table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trips (
                trip_id TEXT PRIMARY KEY,
                route_id TEXT NOT NULL,
                service_id TEXT NOT NULL,
                trip_headsign TEXT,
                trip_short_name TEXT,
                direction_id INTEGER,
                block_id TEXT,
                shape_id TEXT,
                FOREIGN KEY (route_id) REFERENCES routes(route_id),
                FOREIGN KEY (service_id) REFERENCES calendar(service_id)
            )
        """)
        
        # Stop times table (largest table)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stop_times (
                trip_id TEXT NOT NULL,
                arrival_time TEXT,
                departure_time TEXT,
                stop_id TEXT NOT NULL,
                stop_sequence INTEGER NOT NULL,
                stop_headsign TEXT,
                pickup_type INTEGER,
                drop_off_type INTEGER,
                PRIMARY KEY (trip_id, stop_sequence),
                FOREIGN KEY (trip_id) REFERENCES trips(trip_id),
                FOREIGN KEY (stop_id) REFERENCES stops(stop_id)
            )
        """)
        
        # Shapes table (optional)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS shapes (
                shape_id TEXT NOT NULL,
                shape_pt_lat REAL NOT NULL,
                shape_pt_lon REAL NOT NULL,
                shape_pt_sequence INTEGER NOT NULL,
                shape_dist_traveled REAL,
                PRIMARY KEY (shape_id, shape_pt_sequence)
            )
        """)
    
    def _create_indexes(self, cursor: sqlite3.Cursor) -> None:
        """Creates indexes for common query patterns."""
        indexes = [
            # Stop times indexes (most queried)
            "CREATE INDEX IF NOT EXISTS idx_stop_times_stop ON stop_times(stop_id)",
            "CREATE INDEX IF NOT EXISTS idx_stop_times_trip ON stop_times(trip_id)",
            "CREATE INDEX IF NOT EXISTS idx_stop_times_departure ON stop_times(departure_time)",
            
            # Trips indexes
            "CREATE INDEX IF NOT EXISTS idx_trips_route ON trips(route_id)",
            "CREATE INDEX IF NOT EXISTS idx_trips_service ON trips(service_id)",
            
            # Stops indexes
            "CREATE INDEX IF NOT EXISTS idx_stops_name ON stops(stop_name)",
            "CREATE INDEX IF NOT EXISTS idx_stops_coords ON stops(stop_lat, stop_lon)",
            
            # Routes indexes
            "CREATE INDEX IF NOT EXISTS idx_routes_name ON routes(route_short_name)",
            
            # Calendar dates index
            "CREATE INDEX IF NOT EXISTS idx_calendar_dates_date ON calendar_dates(date)",
        ]
        
        for idx_sql in indexes:
            try:
                cursor.execute(idx_sql)
            except sqlite3.Error as e:
                logger.warning(f"Index creation warning: {e}")
    
    def _insert_agency(self, cursor: sqlite3.Cursor, reader: csv.DictReader) -> None:
        """Inserts agency data."""
        for row in reader:
            cursor.execute("""
                INSERT OR REPLACE INTO agency 
                (agency_id, agency_name, agency_url, agency_timezone, agency_lang, agency_phone)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                row.get('agency_id', 'CP'),
                row.get('agency_name', 'CP - Comboios de Portugal'),
                row.get('agency_url', 'https://www.cp.pt'),
                row.get('agency_timezone', 'Europe/Lisbon'),
                row.get('agency_lang', 'pt'),
                row.get('agency_phone', '')
            ))
    
    def _insert_calendar(self, cursor: sqlite3.Cursor, reader: csv.DictReader) -> None:
        """Inserts calendar data."""
        for row in reader:
            cursor.execute("""
                INSERT OR REPLACE INTO calendar
                (service_id, monday, tuesday, wednesday, thursday, friday, saturday, sunday, start_date, end_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row.get('service_id'),
                int(row.get('monday', 0)),
                int(row.get('tuesday', 0)),
                int(row.get('wednesday', 0)),
                int(row.get('thursday', 0)),
                int(row.get('friday', 0)),
                int(row.get('saturday', 0)),
                int(row.get('sunday', 0)),
                row.get('start_date'),
                row.get('end_date')
            ))
    
    def _insert_calendar_dates(self, cursor: sqlite3.Cursor, reader: csv.DictReader) -> None:
        """Inserts calendar dates (exceptions)."""
        for row in reader:
            cursor.execute("""
                INSERT OR REPLACE INTO calendar_dates (service_id, date, exception_type)
                VALUES (?, ?, ?)
            """, (
                row.get('service_id'),
                row.get('date'),
                int(row.get('exception_type', 1))
            ))
    
    def _insert_routes(self, cursor: sqlite3.Cursor, reader: csv.DictReader) -> None:
        """Inserts routes data."""
        for row in reader:
            cursor.execute("""
                INSERT OR REPLACE INTO routes
                (route_id, agency_id, route_short_name, route_long_name, route_desc, route_type, route_color, route_text_color)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row.get('route_id'),
                row.get('agency_id', 'CP'),
                row.get('route_short_name', ''),
                row.get('route_long_name', ''),
                row.get('route_desc', ''),
                int(row.get('route_type', 2)),  # 2 = Rail
                row.get('route_color', ''),
                row.get('route_text_color', '')
            ))
    
    def _insert_stops(self, cursor: sqlite3.Cursor, reader: csv.DictReader) -> None:
        """Inserts stops data."""
        for row in reader:
            try:
                lat = float(row.get('stop_lat', 0)) if row.get('stop_lat') else None
                lon = float(row.get('stop_lon', 0)) if row.get('stop_lon') else None
            except ValueError:
                lat, lon = None, None
            
            cursor.execute("""
                INSERT OR REPLACE INTO stops
                (stop_id, stop_code, stop_name, stop_desc, stop_lat, stop_lon, zone_id, stop_url, location_type, parent_station, platform_code)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row.get('stop_id'),
                row.get('stop_code', ''),
                row.get('stop_name', ''),
                row.get('stop_desc', ''),
                lat,
                lon,
                row.get('zone_id', ''),
                row.get('stop_url', ''),
                int(row.get('location_type', 0)) if row.get('location_type') else 0,
                row.get('parent_station', ''),
                row.get('platform_code', '')
            ))
    
    def _insert_trips(self, cursor: sqlite3.Cursor, reader: csv.DictReader) -> None:
        """Inserts trips data."""
        for row in reader:
            cursor.execute("""
                INSERT OR REPLACE INTO trips
                (trip_id, route_id, service_id, trip_headsign, trip_short_name, direction_id, block_id, shape_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row.get('trip_id'),
                row.get('route_id'),
                row.get('service_id'),
                row.get('trip_headsign', ''),
                row.get('trip_short_name', ''),
                int(row.get('direction_id', 0)) if row.get('direction_id') else 0,
                row.get('block_id', ''),
                row.get('shape_id', '')
            ))
    
    def _insert_stop_times(self, cursor: sqlite3.Cursor, reader: csv.DictReader) -> None:
        """Inserts stop times data (batch insert for performance)."""
        batch = []
        batch_size = 10000
        
        for row in reader:
            batch.append((
                row.get('trip_id'),
                row.get('arrival_time'),
                row.get('departure_time'),
                row.get('stop_id'),
                int(row.get('stop_sequence', 0)),
                row.get('stop_headsign', ''),
                int(row.get('pickup_type', 0)) if row.get('pickup_type') else 0,
                int(row.get('drop_off_type', 0)) if row.get('drop_off_type') else 0
            ))
            
            if len(batch) >= batch_size:
                cursor.executemany("""
                    INSERT OR REPLACE INTO stop_times
                    (trip_id, arrival_time, departure_time, stop_id, stop_sequence, stop_headsign, pickup_type, drop_off_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, batch)
                batch = []
        
        # Insert remaining
        if batch:
            cursor.executemany("""
                INSERT OR REPLACE INTO stop_times
                (trip_id, arrival_time, departure_time, stop_id, stop_sequence, stop_headsign, pickup_type, drop_off_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, batch)
    
    def _insert_shapes(self, cursor: sqlite3.Cursor, reader: csv.DictReader) -> None:
        """Inserts shapes data (batch insert for performance)."""
        batch = []
        batch_size = 10000
        
        for row in reader:
            try:
                lat = float(row.get('shape_pt_lat', 0))
                lon = float(row.get('shape_pt_lon', 0))
                seq = int(row.get('shape_pt_sequence', 0))
                dist = float(row.get('shape_dist_traveled', 0)) if row.get('shape_dist_traveled') else None
            except ValueError:
                continue
            
            batch.append((
                row.get('shape_id'),
                lat,
                lon,
                seq,
                dist
            ))
            
            if len(batch) >= batch_size:
                cursor.executemany("""
                    INSERT OR REPLACE INTO shapes
                    (shape_id, shape_pt_lat, shape_pt_lon, shape_pt_sequence, shape_dist_traveled)
                    VALUES (?, ?, ?, ?, ?)
                """, batch)
                batch = []
        
        if batch:
            cursor.executemany("""
                INSERT OR REPLACE INTO shapes
                (shape_id, shape_pt_lat, shape_pt_lon, shape_pt_sequence, shape_dist_traveled)
                VALUES (?, ?, ?, ?, ?)
            """, batch)
    
    def ensure_database(self, force_refresh: bool = False) -> bool:
        """
        Ensures GTFS database is available and up-to-date.
        
        Args:
            force_refresh: Force download even if cache is valid.
            
        Returns:
            True if database is ready, False otherwise.
        """
        # Check if database exists
        if not force_refresh and self.db_path.exists():
            if not self.check_for_updates():
                logger.info(f"Using existing GTFS database: {self.db_path}")
                return True
        
        # Download and convert
        if self.download_gtfs():
            return self.convert_to_sqlite()
        
        # If download failed but database exists, use existing
        if self.db_path.exists():
            logger.warning("Download failed, using existing database")
            return True
        
        return False
    
    def get_db_connection(self) -> Optional[sqlite3.Connection]:
        """
        Gets a connection to the GTFS SQLite database.
        
        Returns:
            SQLite connection or None if not available.
        """
        if not self.db_path.exists():
            if not self.ensure_database():
                return None
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            return None
    
    def get_active_services(self, date: Optional[datetime] = None) -> List[str]:
        """
        Gets service IDs active on the given date.
        
        Args:
            date: Date to check (defaults to today).
            
        Returns:
            List of active service IDs.
        """
        if date is None:
            date = datetime.now()
        
        date_str = date.strftime('%Y%m%d')
        weekday = date.strftime('%A').lower()
        
        conn = self.get_db_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            
            # Get services from calendar that are active today
            query = f"""
                SELECT service_id FROM calendar
                WHERE start_date <= ? AND end_date >= ?
                AND {weekday} = 1
            """
            cursor.execute(query, (date_str, date_str))
            active_services = set(row['service_id'] for row in cursor.fetchall())
            
            # Add services with exception_type = 1 (service added)
            cursor.execute("""
                SELECT service_id FROM calendar_dates
                WHERE date = ? AND exception_type = 1
            """, (date_str,))
            for row in cursor.fetchall():
                active_services.add(row['service_id'])
            
            # Remove services with exception_type = 2 (service removed)
            cursor.execute("""
                SELECT service_id FROM calendar_dates
                WHERE date = ? AND exception_type = 2
            """, (date_str,))
            for row in cursor.fetchall():
                active_services.discard(row['service_id'])
            
            conn.close()
            return list(active_services)
            
        except sqlite3.Error as e:
            logger.error(f"Error getting active services: {e}")
            conn.close()
            return []


# Global GTFS manager instance
_gtfs_manager: Optional[CPGTFSManager] = None


def get_gtfs_manager() -> CPGTFSManager:
    """Gets or creates the global GTFS manager instance."""
    global _gtfs_manager
    if _gtfs_manager is None:
        _gtfs_manager = CPGTFSManager()
    return _gtfs_manager


# ==========================================================================
# Helper Functions
# ==========================================================================

def _is_cache_valid(last_load: Optional[datetime]) -> bool:
    """Checks if the cache is still valid (not expired)."""
    if last_load is None:
        return False
    hours_elapsed = (datetime.now() - last_load).total_seconds() / 3600
    return hours_elapsed < CACHE_EXPIRATION_HOURS


def fetch_json_with_retry(url: str, timeout: int = REQUEST_TIMEOUT) -> Optional[Any]:
    """
    Fetches JSON data from a URL with retry logic.
    
    Args:
        url: URL to fetch from.
        timeout: Request timeout in seconds.
        
    Returns:
        JSON data if successful, None otherwise.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            wait_time = BACKOFF_FACTOR ** attempt
            logger.warning(f"Timeout. Retrying in {wait_time}s...")
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait_time)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_FACTOR ** attempt)
        except ValueError:
            logger.error("Invalid JSON response")
            return None
    return None


# ==========================================================================
# Station Cache Functions (Real-time API)
# ==========================================================================

def load_cp_aml_stations(force_reload: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Loads CP train stations in the AML (Área Metropolitana de Lisboa) into cache.
    
    Filters the ~462 CP stations to only include the ~81 stations within
    the Lisbon Metropolitan Area bounding box.
    
    Args:
        force_reload: Force refresh even if cache is valid.
        
    Returns:
        Dictionary mapping station code to station info.
    """
    global _cp_stations_cache, _cp_stations_last_load
    
    if not force_reload and _cp_stations_cache and _is_cache_valid(_cp_stations_last_load):
        logger.info(f"Using cached CP AML stations ({len(_cp_stations_cache)} stations)")
        return _cp_stations_cache
    
    logger.info("Loading CP AML stations from API...")
    
    try:
        response = requests.get(CP_STATIONS_URL, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        all_stations = data.get('stations', [])
        
        if not isinstance(all_stations, list):
            logger.error("Unexpected response format from CP stations API")
            return _cp_stations_cache or {}
        
        # Filter to AML region only
        aml_stations = {}
        for station in all_stations:
            try:
                lat = float(station.get('latitude', 0))
                lon = float(station.get('longitude', 0))
                
                # Check if within AML bounding box
                if (AML_BOUNDS['lat_min'] <= lat <= AML_BOUNDS['lat_max'] and
                        AML_BOUNDS['lon_min'] <= lon <= AML_BOUNDS['lon_max']):
                    
                    code = station.get('code', '')
                    aml_stations[code] = {
                        'code': code,
                        'name': station.get('designation', 'Unknown'),
                        'lat': lat,
                        'lon': lon,
                        'railways': station.get('railways', [])
                    }
            except (ValueError, TypeError):
                continue
        
        # Update cache
        _cp_stations_cache = aml_stations
        _cp_stations_last_load = datetime.now()
        
        logger.info(f"\033[1;32m✅ Loaded {len(aml_stations)} CP AML stations\033[0m")
        return aml_stations
        
    except requests.exceptions.Timeout:
        logger.error("Timeout loading CP stations (15s)")
        return _cp_stations_cache or {}
    except requests.exceptions.RequestException as e:
        logger.error(f"Error loading CP stations: {e}")
        return _cp_stations_cache or {}
    except Exception as e:
        logger.error(f"Unexpected error loading CP stations: {e}")
        return _cp_stations_cache or {}


def get_cp_aml_trains() -> List[Dict[str, Any]]:
    """
    Gets real-time train data filtered to only trains serving the AML region.
    
    A train serves the AML if:
    - Its origin station is in the AML, OR
    - Its destination station is in the AML, OR
    - Its current location (lastStation) is in the AML
    
    Returns:
        List of trains serving the AML with full details.
    """
    aml_stations = load_cp_aml_stations()
    aml_codes = set(aml_stations.keys())
    
    if not aml_codes:
        logger.warning("No AML stations loaded, returning all trains")
        data = fetch_json_with_retry(CP_VEHICLES_URL)
        return data.get('vehicles', []) if data else []
    
    try:
        response = requests.get(CP_VEHICLES_URL, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        all_trains = data.get('vehicles', [])
        
        # Filter to trains serving AML
        aml_trains = []
        for train in all_trains:
            origin_code = train.get('origin', {}).get('code', '')
            dest_code = train.get('destination', {}).get('code', '')
            last_station = train.get('lastStation', '')
            
            if (origin_code in aml_codes or
                    dest_code in aml_codes or
                    last_station in aml_codes):
                aml_trains.append(train)
        
        logger.info(f"Filtered to {len(aml_trains)}/{len(all_trains)} trains serving AML")
        return aml_trains
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching trains: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching trains: {e}")
        return []


def search_cp_station(query: str) -> List[Dict[str, Any]]:
    """
    Searches for CP train stations in the AML by name.
    
    Args:
        query: Station name or partial name to search for.
        
    Returns:
        List of matching stations with code, name, lat, lon.
    """
    aml_stations = load_cp_aml_stations()
    query_lower = query.lower().strip()
    
    matches = []
    for code, station in aml_stations.items():
        if query_lower in station['name'].lower():
            matches.append(station)
    
    # Sort by relevance
    matches.sort(key=lambda x: (
        0 if query_lower == x['name'].lower() else 1,
        x['name']
    ))
    
    return matches


# ==========================================================================
# GTFS Query Functions
# ==========================================================================

def get_gtfs_stops_in_aml() -> List[Dict[str, Any]]:
    """
    Gets all GTFS stops within the AML region.
    
    Returns:
        List of stops with id, name, lat, lon.
    """
    manager = get_gtfs_manager()
    conn = manager.get_db_connection()
    
    if not conn:
        return []
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT stop_id, stop_name, stop_lat, stop_lon, stop_code
            FROM stops
            WHERE stop_lat BETWEEN ? AND ?
            AND stop_lon BETWEEN ? AND ?
            AND location_type = 0
            ORDER BY stop_name
        """, (
            AML_BOUNDS['lat_min'], AML_BOUNDS['lat_max'],
            AML_BOUNDS['lon_min'], AML_BOUNDS['lon_max']
        ))
        
        stops = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return stops
        
    except sqlite3.Error as e:
        logger.error(f"Error querying GTFS stops: {e}")
        conn.close()
        return []


def get_gtfs_routes() -> List[Dict[str, Any]]:
    """
    Gets all GTFS routes.
    
    Returns:
        List of routes with id, name, type.
    """
    manager = get_gtfs_manager()
    conn = manager.get_db_connection()
    
    if not conn:
        return []
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT route_id, route_short_name, route_long_name, route_type, route_color
            FROM routes
            ORDER BY route_short_name
        """)
        
        routes = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return routes
        
    except sqlite3.Error as e:
        logger.error(f"Error querying GTFS routes: {e}")
        conn.close()
        return []


def get_stop_departures(
    stop_id: str,
    limit: int = 10,
    date: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Gets upcoming departures from a stop.
    
    Args:
        stop_id: GTFS stop ID.
        limit: Maximum number of departures to return.
        date: Date for schedule (defaults to today).
        
    Returns:
        List of departures with time, route, headsign.
    """
    manager = get_gtfs_manager()
    conn = manager.get_db_connection()
    
    if not conn:
        return []
    
    if date is None:
        date = datetime.now()
    
    # Get active services for the date
    active_services = manager.get_active_services(date)
    
    if not active_services:
        logger.warning("No active services for the specified date")
        conn.close()
        return []
    
    try:
        cursor = conn.cursor()
        
        # Current time in GTFS format (HH:MM:SS)
        current_time = date.strftime('%H:%M:%S')
        
        # Build placeholders for service IDs
        placeholders = ','.join(['?' for _ in active_services])
        
        query = f"""
            SELECT st.departure_time, st.stop_headsign, t.trip_headsign,
                   r.route_short_name, r.route_long_name, r.route_id, t.trip_id
            FROM stop_times st
            JOIN trips t ON st.trip_id = t.trip_id
            JOIN routes r ON t.route_id = r.route_id
            WHERE st.stop_id = ?
            AND t.service_id IN ({placeholders})
            AND st.departure_time >= ?
            AND st.stop_sequence < (
                SELECT MAX(st2.stop_sequence)
                FROM stop_times st2
                WHERE st2.trip_id = st.trip_id
            )
            ORDER BY st.departure_time
            LIMIT ?
        """
        
        cursor.execute(query, [stop_id] + active_services + [current_time, limit])
        
        departures = []
        for row in cursor.fetchall():
            departures.append({
                'departure_time': row['departure_time'],
                'headsign': row['stop_headsign'] or row['trip_headsign'],
                'route_name': row['route_short_name'] or row['route_long_name'],
                'route_id': row['route_id'],
                'trip_id': row['trip_id']
            })
        
        conn.close()
        return departures
        
    except sqlite3.Error as e:
        logger.error(f"Error querying departures: {e}")
        conn.close()
        return []


def get_trip_stops(trip_id: str) -> List[Dict[str, Any]]:
    """
    Gets all stops for a trip in order.
    
    Args:
        trip_id: GTFS trip ID.
        
    Returns:
        List of stops with times and sequence.
    """
    manager = get_gtfs_manager()
    conn = manager.get_db_connection()
    
    if not conn:
        return []
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT st.stop_sequence, st.arrival_time, st.departure_time,
                   s.stop_id, s.stop_name, s.stop_lat, s.stop_lon
            FROM stop_times st
            JOIN stops s ON st.stop_id = s.stop_id
            WHERE st.trip_id = ?
            ORDER BY st.stop_sequence
        """, (trip_id,))
        
        stops = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return stops
        
    except sqlite3.Error as e:
        logger.error(f"Error querying trip stops: {e}")
        conn.close()
        return []


def search_gtfs_stop(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Searches for GTFS stops by name within AML.
    
    Args:
        query: Stop name or partial name.
        limit: Maximum results.
        
    Returns:
        List of matching stops.
    """
    manager = get_gtfs_manager()
    conn = manager.get_db_connection()
    
    if not conn:
        return []
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT stop_id, stop_name, stop_lat, stop_lon, stop_code
            FROM stops
            WHERE stop_name LIKE ?
            AND stop_lat BETWEEN ? AND ?
            AND stop_lon BETWEEN ? AND ?
            AND location_type = 0
            ORDER BY 
                CASE 
                    WHEN LOWER(stop_name) = LOWER(?) THEN 0
                    WHEN LOWER(stop_name) LIKE LOWER(? || '%') THEN 1
                    ELSE 2
                END,
                stop_name
            LIMIT ?
        """, (
            f'%{query}%',
            AML_BOUNDS['lat_min'], AML_BOUNDS['lat_max'],
            AML_BOUNDS['lon_min'], AML_BOUNDS['lon_max'],
            query, query,
            limit
        ))
        
        stops = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return stops
        
    except sqlite3.Error as e:
        logger.error(f"Error searching GTFS stops: {e}")
        conn.close()
        return []


def get_routes_at_stop(stop_id: str) -> List[Dict[str, Any]]:
    """
    Gets all routes that serve a specific stop.
    
    Args:
        stop_id: GTFS stop ID.
        
    Returns:
        List of routes serving the stop.
    """
    manager = get_gtfs_manager()
    conn = manager.get_db_connection()
    
    if not conn:
        return []
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT r.route_id, r.route_short_name, r.route_long_name, r.route_type
            FROM routes r
            JOIN trips t ON r.route_id = t.route_id
            JOIN stop_times st ON t.trip_id = st.trip_id
            WHERE st.stop_id = ?
            ORDER BY r.route_short_name
        """, (stop_id,))
        
        routes = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return routes
        
    except sqlite3.Error as e:
        logger.error(f"Error querying routes at stop: {e}")
        conn.close()
        return []


# ==========================================================================
# LangChain Tools
# ==========================================================================

@tool
def get_train_status() -> str:
    """
    Gets real-time status of CP trains serving the Lisbon Metropolitan Area (AML).
    
    This function combines real-time data from comboios.live API with
    GTFS static schedule data for comprehensive train information.
    
    Returns:
        str: List of AML trains with status, delays, and positions.
    """
    aml_trains = get_cp_aml_trains()
    
    if not aml_trains:
        return "❌ Failed to fetch train status. The API may be temporarily unavailable."
    
    aml_stations = load_cp_aml_stations()
    
    response = "🚆 **CP Trains - Lisbon Metropolitan Area (AML)**\n"
    response += "=" * 50 + "\n\n"
    
    # Group trains by service type
    by_service = defaultdict(list)
    
    for train in aml_trains:
        service_name = train.get('service', {}).get('designation', 'Unknown')
        by_service[service_name].append(train)
    
    # Count stats
    total_trains = len(aml_trains)
    delayed_trains = sum(1 for t in aml_trains if (t.get('delay') or 0) > 0)
    
    response += f"📊 **{total_trains} trains** serving AML\n"
    if delayed_trains > 0:
        response += f"⚠️ **{delayed_trains} trains** with delays\n"
    response += "\n"
    
    # Display by service type
    service_order = ['Urbanos Lisboa', 'Regionais', 'Intercidades', 'Alfa Pendular']
    
    for service_name in service_order:
        if service_name not in by_service:
            continue
            
        trains = by_service[service_name]
        trains.sort(key=lambda x: -(x.get('delay') or 0))
        
        service_emoji = {
            'Urbanos Lisboa': '🚈',
            'Regionais': '🚃',
            'Intercidades': '🚄',
            'Alfa Pendular': '🚅'
        }.get(service_name, '🚆')
        
        response += f"\n{service_emoji} **{service_name}** ({len(trains)} trains)\n"
        response += "-" * 40 + "\n"
        
        for train in trains[:5]:
            train_number = train.get('trainNumber', 'N/A')
            delay = train.get('delay') or 0
            status = train.get('status', 'Unknown')
            has_disruptions = train.get('hasDisruptions', False)
            lat = train.get('latitude')
            lon = train.get('longitude')
            
            origin = train.get('origin', {})
            destination = train.get('destination', {})
            origin_name = origin.get('designation', 'N/A') if origin else 'N/A'
            dest_name = destination.get('designation', 'N/A') if destination else 'N/A'
            
            # Delay in seconds, convert to minutes
            delay_minutes = delay // 60 if delay else 0
            if delay_minutes == 0:
                delay_str = "✅ On time"
            elif delay_minutes > 0:
                delay_str = f"⚠️ {delay_minutes} min late"
            else:
                delay_str = "✅ Ahead"
            
            status_emoji = {
                'IN_TRANSIT': '🚆',
                'AT_STATION': '🚉',
                'STOPPED': '⏸️'
            }.get(status, '❓')
            
            response += f"\n   {status_emoji} **#{train_number}**: {origin_name} → {dest_name}\n"
            response += f"      {delay_str}"
            
            if has_disruptions:
                response += " | ⚠️ Disruptions"
            
            if lat and lon:
                try:
                    response += f"\n      📍 Position: ({float(lat):.4f}, {float(lon):.4f})"
                except (ValueError, TypeError):
                    pass
            
            response += "\n"
        
        if len(trains) > 5:
            response += f"\n   ... and {len(trains) - 5} more {service_name} trains.\n"
    
    response += "\n" + "-" * 50 + "\n"
    response += f"📍 **AML Coverage**: {len(aml_stations)} stations\n"
    response += "🔗 Lines: Cascais, Sintra, Azambuja, Fertagus\n"
    response += "💡 Podes perguntar por uma estação específica para mais detalhes.\n"
    
    return response


@tool
def search_cp_stations(query: str) -> str:
    """
    Searches for CP train stations in the Lisbon Metropolitan Area (AML).
    
    The AML includes ~81 stations across multiple lines:
    - Linha de Cascais (Cais do Sodré ↔ Cascais)
    - Linha de Sintra (Rossio/Oriente ↔ Sintra)
    - Linha de Azambuja (Santa Apolónia/Oriente ↔ Azambuja)
    - Fertagus (Entrecampos ↔ Setúbal)
    
    Args:
        query: Station name or partial name to search for.
        
    Returns:
        str: List of matching stations with details.
    """
    # Try GTFS database first
    gtfs_matches = search_gtfs_stop(query, limit=15)
    
    # Fall back to real-time API if GTFS not available
    if not gtfs_matches:
        rt_matches = search_cp_station(query)
        if not rt_matches:
            return (f"❌ No CP stations found matching '{query}' in the AML region.\n\n"
                    "💡 Try searching for: Oriente, Rossio, Cais do Sodré, Cascais, Sintra, Entrecampos")
        
        response = f"🚉 **CP Stations matching '{query}'** ({len(rt_matches)} found)\n"
        response += "=" * 50 + "\n\n"
        
        for i, station in enumerate(rt_matches[:10], 1):
            name = station.get('name', 'Unknown')
            code = station.get('code', '')
            lat = station.get('lat', 0)
            lon = station.get('lon', 0)
            
            response += f"{i}. **{name}** ({code})\n"
            response += f"   📍 ({lat:.4f}, {lon:.4f})\n\n"
        
        return response
    
    response = f"🚉 **CP Stations matching '{query}'** ({len(gtfs_matches)} found)\n"
    response += "=" * 50 + "\n\n"
    
    for i, stop in enumerate(gtfs_matches[:10], 1):
        name = stop.get('stop_name', 'Unknown')
        stop_id = stop.get('stop_id', '')
        lat = stop.get('stop_lat', 0)
        lon = stop.get('stop_lon', 0)
        
        response += f"{i}. **{name}**\n"
        response += f"   🆔 {stop_id}\n"
        response += f"   📍 ({lat:.4f}, {lon:.4f})\n"
        
        # Get routes at this stop
        routes = get_routes_at_stop(stop_id)
        if routes:
            route_names = list(dict.fromkeys(
                r.get('route_short_name') or r.get('route_long_name') for r in routes
            ))
            route_names = [n for n in route_names if n]
            response += f"   🚆 Routes: {', '.join(route_names[:8])}\n"
        
        response += "\n"
    
    if len(gtfs_matches) > 10:
        response += f"... and {len(gtfs_matches) - 10} more stations.\n"
    
    return response


@tool
def get_train_schedule(station_name: str, limit: int = 10) -> str:
    """
    Gets upcoming train departures from a CP station using GTFS schedule data.
    
    This shows the static schedule. For real-time delays, use get_train_status.
    
    Args:
        station_name: Station name to search for.
        limit: Maximum number of departures to show (default 10).
        
    Returns:
        str: Upcoming departures with times and destinations.
    """
    # Find the station
    stops = search_gtfs_stop(station_name, limit=5)
    
    if not stops:
        return (f"❌ Station '{station_name}' not found.\n\n"
                "💡 Try searching for: Oriente, Rossio, Cais do Sodré, Cascais, Sintra")
    
    # Use the first (best) match
    station = stops[0]
    stop_id = station['stop_id']
    stop_name = station['stop_name']
    
    # Get departures
    departures = get_stop_departures(stop_id, limit=limit)
    
    if not departures:
        return (f"❌ No scheduled departures found for **{stop_name}** today.\n\n"
                "This may be due to:\n"
                "- No more trains today\n"
                "- Holiday schedule\n"
                "- GTFS data not yet available")
    
    now = datetime.now()
    response = f"🚆 **Departures from {stop_name}**\n"
    response += f"📅 {now.strftime('%A, %d %B %Y')}\n"
    response += "=" * 50 + "\n\n"
    
    for dep in departures:
        dep_time = dep['departure_time']
        headsign = dep['headsign'] or 'N/A'
        route_name = dep['route_name'] or ''
        
        # Format time nicely
        try:
            parts = dep_time.split(':')
            hour = int(parts[0]) % 24  # Handle times > 24:00
            minute = parts[1]
            time_str = f"{hour:02d}:{minute}"
        except (IndexError, ValueError):
            time_str = dep_time
        
        response += f"🕐 **{time_str}** → {headsign}\n"
        if route_name:
            response += f"   🚆 {route_name}\n"
        response += "\n"
    
    response += "-" * 50 + "\n"
    response += "💡 Podes perguntar por atrasos em tempo real de um comboio específico.\n"
    
    return response


@tool
def get_cp_routes() -> str:
    """
    Gets all CP train routes/lines available in the GTFS data.
    
    Returns:
        str: List of CP routes with names and types.
    """
    routes = get_gtfs_routes()
    
    if not routes:
        # Fall back to static list
        response = "🚆 **CP Lines - Lisbon Metropolitan Area**\n"
        response += "=" * 50 + "\n\n"
        
        for line_id, info in CP_LINES.items():
            response += f"**{info['name']}**\n"
            response += f"   📍 {info['terminal_a']} ↔ {info['terminal_b']}\n"
            response += f"   📝 {info['description']}\n\n"
        
        return response
    
    response = "🚆 **CP Routes from GTFS Data**\n"
    response += "=" * 50 + "\n\n"
    
    # Deduplicate routes by short name (GTFS has multiple entries per route)
    seen = set()
    unique_routes = []
    for route in routes:
        key = route['route_short_name'] or route['route_id']
        if key not in seen:
            seen.add(key)
            unique_routes.append(route)
    
    # Group by route type
    rail_types = {0: 'Tram', 1: 'Metro', 2: 'Rail', 3: 'Bus', 7: 'Funicular', 11: 'Trolleybus', 12: 'Monorail'}
    
    for route in unique_routes:
        route_type = rail_types.get(route['route_type'], 'Other')
        name = route['route_short_name'] or route['route_long_name'] or route['route_id']
        long_name = route['route_long_name'] or ''
        color = route.get('route_color', '')
        
        response += f"🚆 **{name}**"
        if long_name and long_name != name:
            response += f" - {long_name}"
        response += f"\n   📋 Type: {route_type}\n"
        if color:
            response += f"   🎨 Color: #{color}\n"
        response += "\n"
    
    return response


@tool
def plan_train_trip(origin: str, destination: str) -> str:
    """
    Plans a train trip between two stations using GTFS schedule data.
    
    Calculates travel time based on actual GTFS timetables (NOT estimated).
    Also shows real-time delay information when available.
    
    Args:
        origin: Starting station name (e.g., 'Entrecampos', 'Rossio').
        destination: Ending station name (e.g., 'Sintra', 'Cascais').
        
    Returns:
        str: Trip details including travel time from GTFS, next departures, and delays.
    """
    manager = get_gtfs_manager()
    
    # Find origin station
    origin_stops = search_gtfs_stop(origin, limit=3)
    if not origin_stops:
        return (f"❌ Station '{origin}' not found.\n\n"
                "💡 Try: Oriente, Rossio, Entrecampos, Cais do Sodré, Cascais, Sintra")
    
    # Find destination station
    dest_stops = search_gtfs_stop(destination, limit=3)
    if not dest_stops:
        return (f"❌ Station '{destination}' not found.\n\n"
                "💡 Try: Oriente, Rossio, Entrecampos, Cais do Sodré, Cascais, Sintra")
    
    origin_station = origin_stops[0]
    dest_station = dest_stops[0]
    origin_id = origin_station['stop_id']
    dest_id = dest_station['stop_id']
    origin_name = origin_station['stop_name']
    dest_name = dest_station['stop_name']
    
    now = datetime.now()
    current_time = now.strftime('%H:%M:%S')
    
    # Get active services for today
    active_services = manager.get_active_services(now)
    
    if not active_services:
        return "❌ No train services active today. This might be a holiday or data issue."
    
    conn = manager.get_db_connection()
    if not conn:
        return "❌ Database not available. Try running `initialize_cp_gtfs()` first."
    
    try:
        cursor = conn.cursor()
        placeholders = ','.join(['?' for _ in active_services])
        
        # Find ALL trips that go from origin to destination (no LIMIT)
        # We need all results to show accurate remaining count
        query = f"""
            SELECT 
                st_origin.trip_id,
                st_origin.departure_time as origin_departure,
                st_dest.arrival_time as dest_arrival,
                st_origin.stop_sequence as origin_seq,
                st_dest.stop_sequence as dest_seq,
                t.trip_headsign,
                r.route_short_name,
                r.route_long_name
            FROM stop_times st_origin
            JOIN stop_times st_dest ON st_origin.trip_id = st_dest.trip_id
            JOIN trips t ON st_origin.trip_id = t.trip_id
            JOIN routes r ON t.route_id = r.route_id
            WHERE st_origin.stop_id = ?
            AND st_dest.stop_id = ?
            AND st_origin.stop_sequence < st_dest.stop_sequence
            AND t.service_id IN ({placeholders})
            AND st_origin.departure_time >= ?
            ORDER BY st_origin.departure_time
        """
        
        cursor.execute(query, [origin_id, dest_id] + active_services + [current_time])
        trips = cursor.fetchall()
        conn.close()
        
        if not trips:
            # Try without time constraint to see if route exists at all
            conn = manager.get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) 
                FROM stop_times st_origin
                JOIN stop_times st_dest ON st_origin.trip_id = st_dest.trip_id
                JOIN trips t ON st_origin.trip_id = t.trip_id
                WHERE st_origin.stop_id = ?
                AND st_dest.stop_id = ?
                AND st_origin.stop_sequence < st_dest.stop_sequence
            """, (origin_id, dest_id))
            total_trips = cursor.fetchone()[0]
            conn.close()
            
            if total_trips == 0:
                return (f"❌ No direct train service found between **{origin_name}** and **{dest_name}**.\n\n"
                        "💡 These stations may be on different lines. Consider:\n"
                        "   • Transfer at a connecting station (e.g., Oriente, Entrecampos)\n"
                        "   • Podes perguntar por informação sobre uma estação específica")
            else:
                return (f"⏰ No more trains today from **{origin_name}** to **{dest_name}**.\n\n"
                        f"There are {total_trips} trips on other days. Try again tomorrow or check schedules online.")
        
        # Calculate travel time from GTFS data
        def parse_gtfs_time(time_str: str) -> int:
            """Convert GTFS time (HH:MM:SS) to minutes since midnight."""
            parts = time_str.split(':')
            return int(parts[0]) * 60 + int(parts[1])
        
        def format_time(time_str: str) -> str:
            """Format GTFS time for display (handle times > 24:00)."""
            parts = time_str.split(':')
            hour = int(parts[0]) % 24
            return f"{hour:02d}:{parts[1]}"
        
        # Calculate travel times for ALL trips to find min/max range
        def calc_trip_travel(trip) -> int:
            dep_m = parse_gtfs_time(trip['origin_departure'])
            arr_m = parse_gtfs_time(trip['dest_arrival'])
            diff = arr_m - dep_m
            return diff + 24 * 60 if diff < 0 else diff
        
        trip_durations = [calc_trip_travel(t) for t in trips]
        min_duration = min(trip_durations)
        max_duration = max(trip_durations)
        
        # Get route name: use the MOST COMMON route across all trips
        # (e.g., Oriente->Sintra has 68 trips on "Linha de Sintra" and 14 on
        # "Linha da Azambuja" - we should show "Linha de Sintra")
        from collections import Counter
        route_counts = Counter(
            (t['route_short_name'] or t['route_long_name'] or 'CP') for t in trips
        )
        route_name = route_counts.most_common(1)[0][0]
        distinct_routes = list(route_counts.keys())
        multi_route = len(distinct_routes) > 1
        
        # Get real-time delay info
        aml_trains = get_cp_aml_trains()
        
        # Build map of real-time trains heading to destination, keyed by departure time
        # for matching with GTFS scheduled departures
        realtime_trains = {}
        route_has_delays = False
        route_delay_mins = 0
        for train in aml_trains:
            train_headsign = train.get('destination', {}).get('designation', '')
            if dest_name.lower() in train_headsign.lower() or train_headsign.lower() in dest_name.lower():
                delay = train.get('delay') or 0
                delay_mins = delay // 60 if delay else 0
                train_number = train.get('trainNumber', '')
                realtime_trains[train_number] = {
                    'delay_mins': delay_mins,
                    'headsign': train_headsign,
                    'status': train.get('status', '')
                }
                if delay_mins > 0:
                    route_has_delays = True
                    route_delay_mins = max(route_delay_mins, delay_mins)
        
        # Build response
        response = f"🚆 **Comboio: {origin_name} → {dest_name}**\n"
        response += "=" * 50 + "\n\n"
        
        # Summary box at top
        response += "📊 **RESUMO DA VIAGEM**\n"
        if multi_route:
            routes_str = ", ".join(distinct_routes)
            response += f"   🚆 Linhas: **{routes_str}**\n"
        else:
            response += f"   🚆 Linha: **{route_name}**\n"
        # Show duration range if trips vary, otherwise single value
        if min_duration == max_duration:
            response += f"   ⏱️ Duração: **{min_duration} minutos**\n"
        else:
            response += f"   ⏱️ Duração: **{min_duration}-{max_duration} minutos**\n"
        # Only show delay status if we have real-time info
        if realtime_trains:
            if route_has_delays:
                response += f"   📍 Estado: ⚠️ Alguns comboios com +{route_delay_mins}min atraso\n"
            else:
                response += "   📍 Estado: ✅ Comboios a horas (tempo real)\n"
        else:
            response += "   📍 Estado: ℹ️ Sem dados em tempo real\n"
        response += f"   📊 Partidas restantes hoje: **{len(trips)}**\n"
        response += "\n"
        
        response += "-" * 50 + "\n"
        
        # Show up to 8 departures
        display_count = min(8, len(trips))
        response += f"📋 **Próximas {display_count} Partidas:**\n\n"
        
        for i, trip in enumerate(trips[:8], 1):
            origin_dep = trip['origin_departure']
            dest_arr = trip['dest_arrival']
            trip_route = trip['route_short_name'] or trip['route_long_name'] or 'CP'
            
            # Calculate travel time in minutes from GTFS departure→arrival
            trip_travel_mins = calc_trip_travel(trip)
            
            # Format display times
            dep_display = format_time(origin_dep)
            arr_display = format_time(dest_arr)
            
            response += f"   🕐 **{dep_display}** → {arr_display} ({trip_travel_mins}min)"
            # Show route label per departure if multiple routes
            if multi_route:
                response += f" [{trip_route}]"
            response += "\n"
        
        if len(trips) > 8:
            response += f"\n   ... e mais {len(trips) - 8} partidas restantes hoje.\n"
        
        response += "\n" + "-" * 50 + "\n"
        response += f"📅 {now.strftime('%A, %d %B %Y')} | {now.strftime('%H:%M')}\n"
        response += "💡 Horários: cp.pt | Bilhetes: app CP ou estação\n"
        
        return response
        
    except Exception as e:
        if conn:
            conn.close()
        logger.error(f"Error planning train trip: {e}")
        return f"❌ Error planning trip: {str(e)}"



def initialize_cp_gtfs(force_refresh: bool = False) -> str:
    """
    Initializes or updates the CP GTFS database.
    
    Downloads the latest GTFS feed from CP and converts it to SQLite
    for fast schedule queries. This is automatically called when needed,
    but can be manually triggered to force a refresh.
    
    Args:
        force_refresh: Force download even if data is recent.
        
    Returns:
        str: Status message about the initialization.
    """
    manager = get_gtfs_manager()
    
    response = "🚆 **CP GTFS Database Initialization**\n"
    response += "=" * 50 + "\n\n"
    
    if manager.db_path.exists() and not force_refresh:
        metadata = manager._load_metadata()
        last_download = metadata.get('last_download', 'Unknown')
        response += "📊 **Existing database found**\n"
        response += f"   Last updated: {last_download}\n\n"
        
        if not manager.check_for_updates():
            response += "✅ Database is up-to-date. No refresh needed.\n"
            
            # Show some stats
            conn = manager.get_db_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM stops")
                stops_count = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM routes")
                routes_count = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM trips")
                trips_count = cursor.fetchone()[0]
                conn.close()
                
                response += "\n📈 **Database Stats:**\n"
                response += f"   - Stops: {stops_count}\n"
                response += f"   - Routes: {routes_count}\n"
                response += f"   - Trips: {trips_count}\n"
            
            return response
    
    response += "📥 **Downloading CP GTFS feed...**\n"
    
    if manager.download_gtfs():
        response += "✅ Download successful\n\n"
        
        response += "🔄 **Converting to SQLite...**\n"
        if manager.convert_to_sqlite():
            response += "✅ Conversion successful\n\n"
            
            # Show stats
            conn = manager.get_db_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM stops")
                stops_count = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM routes")
                routes_count = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM trips")
                trips_count = cursor.fetchone()[0]
                conn.close()
                
                response += "📈 **Database Stats:**\n"
                response += f"   - Stops: {stops_count}\n"
                response += f"   - Routes: {routes_count}\n"
                response += f"   - Trips: {trips_count}\n"
            
            response += f"\n💾 Database location: {manager.db_path}\n"
            return response
        else:
            response += "❌ Conversion failed\n"
            return response
    else:
        response += "❌ Download failed\n"
        if manager.db_path.exists():
            response += "⚠️ Using existing database\n"
        return response


@tool
def get_train_frequency(
    route_name: str,
    station_name: Optional[str] = None,
) -> str:
    """
    Estimates train service frequency (headway) for a CP train line.
    Calculates average time between departures by time window.
    Uses GTFS stop_times data since frequencies.txt is not available.

    Args:
        route_name: Train line/route name (e.g., 'Sintra', 'Cascais', 'Azambuja', 'Sado').
        station_name: Optional station name to check frequency at.
                     If not provided, uses the main origin station of the line.

    Returns:
        str: Formatted frequency information by time window.
        
    Examples:
        >>> get_train_frequency("Sintra")
        >>> get_train_frequency("Cascais", station_name="Cais do Sodré")
    """
    manager = CPGTFSManager()
    conn = manager.get_db_connection()
    if not conn:
        return "CP GTFS database unavailable."

    try:
        cursor = conn.cursor()

        # Find route matching name
        cursor.execute(
            "SELECT route_id, route_short_name, route_long_name FROM routes WHERE route_long_name LIKE ? OR route_short_name LIKE ?",
            (f"%{route_name}%", f"%{route_name}%"),
        )
        route = cursor.fetchone()
        if not route:
            conn.close()
            return f"Route '{route_name}' not found in CP data. Try: Sintra, Cascais, Azambuja, Sado"

        route_id = route["route_id"]
        route_display = route["route_long_name"] or route["route_short_name"]

        # Get active services for today
        active_services = manager.get_active_services()
        if not active_services:
            conn.close()
            return "No active train services found for today."

        ph_s = ",".join(["?" for _ in active_services])

        # Determine which station to analyze
        if station_name:
            cursor.execute(
                "SELECT stop_id, stop_name FROM stops WHERE stop_name LIKE ? LIMIT 1",
                (f"%{station_name}%",),
            )
            stop_row = cursor.fetchone()
            if not stop_row:
                conn.close()
                return f"Station '{station_name}' not found."
            stop_id = stop_row["stop_id"]
            stop_display = stop_row["stop_name"]
        else:
            # Use first station on the route
            cursor.execute(
                f"""
                SELECT st.stop_id, s.stop_name
                FROM stop_times st
                JOIN trips t ON st.trip_id = t.trip_id
                JOIN stops s ON st.stop_id = s.stop_id
                WHERE t.route_id = ? AND t.service_id IN ({ph_s})
                ORDER BY st.stop_sequence ASC
                LIMIT 1
                """,
                [route_id] + active_services,
            )
            stop_row = cursor.fetchone()
            if not stop_row:
                conn.close()
                return f"No scheduled trips found for '{route_name}' today."
            stop_id = stop_row["stop_id"]
            stop_display = stop_row["stop_name"]

        # Get all departure times at this station for this route today
        cursor.execute(
            f"""
            SELECT st.departure_time
            FROM stop_times st
            JOIN trips t ON st.trip_id = t.trip_id
            WHERE t.route_id = ? AND st.stop_id = ? AND t.service_id IN ({ph_s})
            ORDER BY st.departure_time ASC
            """,
            [route_id, stop_id] + active_services,
        )

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return f"No departures found for '{route_name}' at '{stop_display}' today."

        # Parse times into minutes since midnight
        departures = []
        for row in rows:
            time_str = row["departure_time"]
            parts = time_str.split(":")
            if len(parts) >= 2:
                h, m = int(parts[0]), int(parts[1])
                departures.append(h * 60 + m)

        departures = sorted(set(departures))

        # Define time windows
        windows = [
            ("🌅 Morning (06:00-09:59)", 360, 600),
            ("☀️ Midday (10:00-13:59)", 600, 840),
            ("🌤️ Afternoon (14:00-17:59)", 840, 1080),
            ("🌙 Evening (18:00-22:59)", 1080, 1380),
            ("🌃 Night (23:00-01:59)", 1380, 1560),
        ]

        response = f"🚆 **Frequência: {route_display}**\n"
        response += f"📍 Estação: {stop_display}\n"
        response += "=" * 50 + "\n\n"
        response += f"📊 Total de comboios hoje: {len(departures)}\n\n"

        for window_name, start_min, end_min in windows:
            window_deps = [d for d in departures if start_min <= d < end_min]

            if len(window_deps) < 2:
                if len(window_deps) == 1:
                    t = window_deps[0]
                    response += f"{window_name}: 1 comboio ({t // 60:02d}:{t % 60:02d})\n"
                else:
                    response += f"{window_name}: Sem serviço\n"
                continue

            # Calculate headways between consecutive departures
            headways = [window_deps[i + 1] - window_deps[i] for i in range(len(window_deps) - 1)]
            avg_headway = sum(headways) / len(headways)
            min_headway = min(headways)
            max_headway = max(headways)

            first_dep = window_deps[0]
            last_dep = window_deps[-1]

            response += f"{window_name}\n"
            response += f"   ⏱️ Frequência média: **{avg_headway:.0f} min**\n"
            response += f"   📏 Min/Max: {min_headway}-{max_headway} min\n"
            response += f"   🕒 Primeiro: {first_dep // 60:02d}:{first_dep % 60:02d} | Último: {last_dep // 60:02d}:{last_dep % 60:02d}\n"
            response += f"   📈 Comboios: {len(window_deps)}\n\n"

        response += "📌 **Fonte:** Dados GTFS CP (horários programados)\n"
        response += "⚠️ Consulte cp.pt para informação em tempo real.\n"

        return response

    except Exception as e:
        logger.error(f"Error calculating train frequency: {e}")
        return f"Error calculating train frequency: {e}"


# ==========================================================================
# Test Block
# ==========================================================================

if __name__ == "__main__":
    import sys

    # Fix Windows console encoding for emojis
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

    print("\n" + "=" * 70)
    print("\033[1m🧪 COMPREHENSIVE TEST: CP Trains API Tools\033[0m")
    print("=" * 70)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    test_results = {"passed": 0, "failed": 0, "total": 0}

    def run_test(test_name: str, test_func, *args, **kwargs):
        """Helper to run tests with error handling."""
        test_results["total"] += 1
        print(f"\n\033[1m{'─' * 70}\033[0m")
        print(f"\033[1;36m🔬 TEST {test_results['total']}: {test_name}\033[0m")
        print(f"{'─' * 70}")
        try:
            result = test_func(*args, **kwargs)
            if isinstance(result, str) and len(result) > 800:
                print(result[:800] + "\n\n... (truncated for readability)")
            elif result is not None:
                print(result)
            test_results["passed"] += 1
            print("\n\033[1;32m✅ PASSED\033[0m")
            return result
        except Exception as e:
            print(f"\n\033[1;31m❌ FAILED: {e}\033[0m")
            test_results["failed"] += 1
            return None

    # =========================================================================
    # GTFS INITIALIZATION
    # =========================================================================
    # TEST 1: Initialize GTFS data
    def _test_gtfs_init():
        return initialize_cp_gtfs(force_refresh=False)

    run_test("GTFS Initialization", _test_gtfs_init)

    # TEST 2: Load AML stations
    def _test_aml_stations():
        stations = load_cp_aml_stations()
        if not stations:
            raise AssertionError("No AML stations loaded")
        return f"Loaded {len(stations)} AML stations from real-time API"

    run_test("Load AML Stations", _test_aml_stations)

    # =========================================================================
    # STATION & SCHEDULE TOOLS
    # =========================================================================
    # TEST 3: Search stations
    run_test(
        "search_cp_stations('Oriente')",
        search_cp_stations.invoke,
        {"query": "Oriente"},
    )

    # TEST 4: Get train status (real-time)
    run_test(
        "get_train_status (Real-Time)",
        get_train_status.invoke,
        {},
    )

    # TEST 5: Get schedule at a station
    run_test(
        "get_train_schedule('Lisboa', limit=5)",
        get_train_schedule.invoke,
        {"station_name": "Lisboa", "limit": 5},
    )

    # TEST 6: Get CP routes
    run_test(
        "get_cp_routes (All AML Lines)",
        get_cp_routes.invoke,
        {},
    )

    # =========================================================================
    # FREQUENCY TOOL TESTS
    # =========================================================================
    # TEST 7: Train frequency - Sintra line
    run_test(
        "get_train_frequency('Sintra')",
        get_train_frequency.invoke,
        {"route_name": "Sintra"},
    )

    # TEST 8: Train frequency - Cascais line
    run_test(
        "get_train_frequency('Cascais')",
        get_train_frequency.invoke,
        {"route_name": "Cascais"},
    )

    # TEST 9: Train frequency with specific station
    run_test(
        "get_train_frequency('Sintra', station='Amadora')",
        get_train_frequency.invoke,
        {"route_name": "Sintra", "station_name": "Amadora"},
    )

    # TEST 10: Frequency output format validation
    def _test_cp_frequency_format():
        result = get_train_frequency.invoke({"route_name": "Sintra"})
        checks = {
            "has_line_name": "Sintra" in result,
            "has_morning": "Morning" in result or "morning" in result.lower(),
            "has_frequency": "min" in result.lower(),
            "has_train_count": "Comboios" in result or "comboios" in result.lower(),
        }
        errors = [k for k, v in checks.items() if not v]
        if errors:
            raise AssertionError(f"Missing in frequency output: {errors}")
        return f"Frequency output format valid (checks: {list(checks.keys())})"

    run_test("Frequency Output Format Validation", _test_cp_frequency_format)

    # TEST 11: Frequency - unknown line (edge case)
    run_test(
        "get_train_frequency('XYZLine') - Invalid Route",
        get_train_frequency.invoke,
        {"route_name": "XYZLine"},
    )

    # =========================================================================
    # TRIP PLANNING TESTS
    # =========================================================================
    # TEST 12: Plan a train trip
    run_test(
        "plan_train_trip: Oriente → Sintra",
        plan_train_trip.invoke,
        {"origin": "Oriente", "destination": "Sintra"},
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("\033[1m📊 TEST SUMMARY\033[0m")
    print("=" * 70)
    print(f"\033[1;32m✅ Passed: {test_results['passed']}/{test_results['total']}\033[0m")
    if test_results["failed"] > 0:
        print(f"\033[1;31m❌ Failed: {test_results['failed']}/{test_results['total']}\033[0m")
    else:
        print("\033[1;32m🎉 ALL TESTS PASSED!\033[0m")
    print("=" * 70)
