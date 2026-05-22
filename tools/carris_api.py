# ==========================================================================
# Master Thesis - Carris API Tools (Urban Lisbon Buses & Trams)
#   - André Filipe Gomes Silvestre, 20240502
#
#   Static GTFS data and real-time vehicle tracking for Carris
#   (Lisbon's urban bus and tram operator, NOT Carris Metropolitana).
#
#   Data Sources:
#     - GTFS Static: https://gateway.carris.pt/gateway/gtfs/api/v2.8/GTFS
#     - GTFS-RT (Real-Time): https://gateway.carris.pt/gateway/gtfs/api/v2.8/GTFS/realtime/vehiclepositions
#
#   Storage:
#     - SQLite database in ./data/carris/carris.db
#     - Update check via HTTP headers (Content-Disposition filename date)
#
#   GTFS-RT Vehicle Positions Feed:
#     - Protocol Buffers format (requires gtfs-realtime-bindings)
#     - Provides: trip_id, route_id, stop_id, position (lat/lon), vehicle_id, license_plate
#     - Links to static GTFS via trip_id, route_id, and stop_id
#
#   Usage:
#       python tools/carris_api.py                  [Runs all tests without rebuilding the database]
#       python tools/carris_api.py --rebuild-db     [Rebuilds the database]
# ==========================================================================

# Required libraries:
# pip install protobuf gtfs-realtime-bindings requests langchain-core

import argparse
import csv
import json
import logging
import os
import re
import sqlite3
import time
import unicodedata
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import requests
from langchain_core.tools import tool
import contextlib

# GTFS-RT Protocol Buffers
try:
    from google.transit import gtfs_realtime_pb2

    GTFS_RT_AVAILABLE = True
except ImportError:
    GTFS_RT_AVAILABLE = False
    gtfs_realtime_pb2 = None

logger = logging.getLogger(__name__)

try:
    from tools.utils import haversine_distance
except ImportError:
    from utils import haversine_distance

try:
    from tools.location_resolver import build_location_ambiguity_preamble, get_location_display_name
    from tools.runtime_paths import resolve_runtime_data_dir, seed_runtime_data_dir
    from tools.transport_release_assets import ensure_runtime_data_from_release
except ImportError:
    from location_resolver import build_location_ambiguity_preamble, get_location_display_name
    from runtime_paths import resolve_runtime_data_dir, seed_runtime_data_dir
    from transport_release_assets import ensure_runtime_data_from_release

# ==========================================================================
# Configuration
# ==========================================================================

# GTFS Static Data
CARRIS_GTFS_URL = "https://gateway.carris.pt/gateway/gtfs/api/v2.8/GTFS"

# GTFS-RT Real-Time Vehicle Positions (Official Carris API - Protocol Buffers)
CARRIS_GTFS_RT_URL = (
    "https://gateway.carris.pt/gateway/gtfs/api/v2.8/GTFS/realtime/vehiclepositions"
)

# Data directory (relative to project root locally, writable runtime path on HF)
SOURCE_CARRIS_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "carris"
_CARRIS_RUNTIME_DATA_DIR = resolve_runtime_data_dir(SOURCE_CARRIS_DATA_DIR, "carris")
seed_runtime_data_dir(SOURCE_CARRIS_DATA_DIR, _CARRIS_RUNTIME_DATA_DIR, ("carris.db", "metadata.json"))
CARRIS_DATA_DIR = str(_CARRIS_RUNTIME_DATA_DIR)
CARRIS_DB_PATH = str(_CARRIS_RUNTIME_DATA_DIR / "carris.db")
CARRIS_METADATA_PATH = str(_CARRIS_RUNTIME_DATA_DIR / "metadata.json")
CARRIS_RUNTIME_RELEASE_ENV_PREFIX = "CARRIS_RUNTIME_RELEASE"
CARRIS_RUNTIME_RELEASE_ASSET = "carris_runtime.zip"

# Request timeout
REQUEST_TIMEOUT = 120  # GTFS download can take time
REALTIME_TIMEOUT = 25  # Real-time requests (GTFS-RT endpoint can be slow)
REALTIME_MAX_RETRIES = 3  # Number of retry attempts for GTFS-RT
REALTIME_RETRY_BACKOFF = 2  # Base backoff in seconds between retries

# Cache for GTFS-RT data (avoid excessive API calls)
_gtfs_rt_cache: Dict[str, Any] = {
    "data": None,
    "timestamp": 0,
    "ttl": 30,  # Cache for 30 seconds
}
_gtfs_rt_feed_meta: Dict[str, Any] = {
    "source": "uninitialized",
    "generated_at": None,
    "data_age_seconds": None,
    "last_error": None,
    "vehicle_count": 0,
}

# Route type mapping (GTFS standard: 0=Tram, 3=Bus)
ROUTE_TYPES = {
    "tram": 0,
    "elétrico": 0,
    "eletrico": 0,
    "bus": 3,
    "autocarro": 3,
}

# Vehicle status mapping (GTFS-RT VehicleStopStatus)
VEHICLE_STATUS = {
    0: "INCOMING_AT",  # Approaching stop
    1: "STOPPED_AT",  # At stop
    2: "IN_TRANSIT_TO",  # In transit to next stop
}

# Nominatim geocoding
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


# ==========================================================================
# Utility Functions
# ==========================================================================


def geocode_location(
    place_name: str,
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Geocodes a place name using the shared Lisbon/AML resolver.

    Args:
        place_name: Place name to geocode.

    Returns:
        Tuple of (latitude, longitude, display_name) or (None, None, None) on error.
    """
    try:
        from tools.location_resolver import resolve_location_query
    except ImportError:
        from location_resolver import resolve_location_query

    try:
        resolved = resolve_location_query(
            place_name,
            prefer_city=True,
            allow_aml=True,
        )
        if resolved.get("success") and resolved.get("lat") is not None and resolved.get("lon") is not None:
            return (
                resolved["lat"],
                resolved["lon"],
                resolved["display_name"],
            )
    except Exception as e:
        logger.warning(f"Geocoding failed for '{place_name}': {e}")

    return None, None, None


def time_str_to_minutes(time_str: str) -> int:
    """
    Convert HH:MM:SS time string to minutes since midnight.
    Handles times > 24:00:00 (GTFS convention for trips past midnight).

    Args:
        time_str: Time in format "HH:MM:SS" or "HH:MM"

    Returns:
        Minutes since midnight
    """
    parts = time_str.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    return hours * 60 + minutes


def _format_service_clock(total_minutes: int) -> str:
    """Format service minutes, marking GTFS after-midnight times clearly."""
    hour = total_minutes // 60
    minute = total_minutes % 60
    if hour >= 24:
        return f"{hour % 24:02d}:{minute:02d} (next day)"
    return f"{hour:02d}:{minute:02d}"


def _minutes_until_clock_time(clock_text: str) -> Optional[int]:
    """Return minutes from now until the next HH:MM clock occurrence."""
    try:
        target_minutes = time_str_to_minutes(clock_text)
    except (ValueError, IndexError):
        return None
    now = datetime.now()
    current_minutes = now.hour * 60 + now.minute
    delta = target_minutes - current_minutes
    if delta < 0:
        delta += 24 * 60
    return delta


def minutes_to_time_str(minutes: int) -> str:
    """
    Convert minutes since midnight to HH:MM format.

    Args:
        minutes: Minutes since midnight

    Returns:
        Time string in HH:MM format
    """
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}"


def _normalize_carris_text(text: str) -> str:
    """Normalizes Carris stop/headsign text for accent-insensitive comparisons."""
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9\s/-]", " ", normalized)
    replacements = {
        r"\bpca\b": "praca",
        r"\bpc\b": "praca",
        r"\bpr\b": "praca",
        r"\blg\b": "largo",
        r"\bcalc\b": "calcada",
        r"\bcc\b": "calcada",
        r"\bav\b": "avenida",
        r"\br\b": "rua",
        r"\bhosp\b": "hospital",
        r"\best\b": "estacao",
    }
    for pattern, replacement in replacements.items():
        normalized = re.sub(pattern, replacement, normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _clean_carris_headsign(text: Optional[str]) -> str:
    """Cleans raw Carris headsign/route text for user-facing output."""
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+[-/]\s*$", "", cleaned)
    cleaned = re.sub(r"\($", "", cleaned).strip()
    if cleaned.count("(") > cleaned.count(")"):
        cleaned = cleaned.rstrip("(").strip()
    return cleaned


def _truncate_display_text(text: str, max_length: int = 42) -> str:
    """Truncates user-facing text without leaving dangling punctuation or parentheses."""
    cleaned = _clean_carris_headsign(text)
    if len(cleaned) <= max_length:
        return cleaned
    trimmed = cleaned[: max_length - 1].rstrip(" -/(")
    return f"{trimmed}…"


def _resolve_carris_headsign(
    trip_headsign: Optional[str],
    route_long_name: Optional[str],
    direction_id: Optional[int],
) -> str:
    """Resolves the best user-facing destination label for a Carris trip."""
    cleaned_trip_headsign = _clean_carris_headsign(trip_headsign)
    cleaned_route_long_name = _clean_carris_headsign(route_long_name)

    if cleaned_trip_headsign and " - " not in cleaned_trip_headsign:
        return cleaned_trip_headsign

    parts = [part.strip() for part in cleaned_route_long_name.split(" - ") if part.strip()]
    if len(parts) == 2 and direction_id in {0, 1}:
        return parts[1] if direction_id == 0 else parts[0]

    return cleaned_trip_headsign or cleaned_route_long_name or "Unknown"


def _update_gtfs_rt_feed_meta(
    *,
    source: str,
    generated_at: Optional[datetime] = None,
    data_age_seconds: Optional[float] = None,
    last_error: Optional[str] = None,
    vehicle_count: int = 0,
) -> None:
    """Stores runtime metadata about the Carris GTFS-RT vehicle feed."""
    _gtfs_rt_feed_meta.update(
        {
            "source": source,
            "generated_at": generated_at.isoformat() if isinstance(generated_at, datetime) else None,
            "data_age_seconds": data_age_seconds,
            "last_error": last_error,
            "vehicle_count": vehicle_count,
        }
    )


def _build_gtfs_rt_freshness_note() -> str:
    """Builds a short freshness note for Carris GTFS-RT powered outputs."""
    source = _gtfs_rt_feed_meta.get("source")
    age = _gtfs_rt_feed_meta.get("data_age_seconds")
    last_error = _gtfs_rt_feed_meta.get("last_error")

    if source == "live":
        line = "📡 Carris GTFS-RT: live vehicle feed active."
    elif source == "cache":
        age_text = f"{int(age)}s" if age is not None else "unknown age"
        line = f"📡 Carris GTFS-RT: cached live snapshot in use ({age_text} old)."
    elif source == "stale_cache":
        age_text = f"{int(age)}s" if age is not None else "unknown age"
        line = f"⚠️ Carris GTFS-RT: using stale cached vehicle data ({age_text} old) because the live feed is temporarily unavailable."
    elif source == "unavailable":
        line = "⚠️ Carris GTFS-RT: live vehicle data is temporarily unavailable."
    else:
        line = ""

    if last_error and source in {"stale_cache", "unavailable"}:
        line = f"{line} Last feed issue: {last_error}".strip()

    return line.strip()


def _search_stop_rows(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 10,
) -> List[sqlite3.Row]:
    """Searches Carris stops with SQL first and accent-insensitive fallback second."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT stop_id, stop_name, stop_lat, stop_lon, stop_code FROM stops WHERE stop_name LIKE ? ORDER BY stop_name LIMIT ?",
        (f"%{query}%", limit),
    )
    rows = cursor.fetchall()
    if rows:
        return rows

    normalized_query = _normalize_carris_text(query)
    if not normalized_query:
        return []

    stopwords = {"de", "da", "do", "das", "dos", "e"}
    query_tokens = [token for token in normalized_query.split() if token not in stopwords]
    cursor.execute(
        "SELECT stop_id, stop_name, stop_lat, stop_lon, stop_code FROM stops"
    )

    scored_rows: List[Tuple[int, sqlite3.Row]] = []
    for row in cursor.fetchall():
        normalized_name = _normalize_carris_text(row["stop_name"])
        if not normalized_name:
            continue

        score = 0
        if normalized_query == normalized_name:
            score = 100
        elif normalized_query in normalized_name:
            score = 80
        elif query_tokens and all(token in normalized_name for token in query_tokens):
            score = 60 + len(query_tokens)
        else:
            continue

        scored_rows.append((score, row))

    scored_rows.sort(key=lambda item: (-item[0], item[1]["stop_name"]))
    return [row for _, row in scored_rows[:limit]]


def _format_delay_label(delay_mins: int) -> str:
    """Formats a delay indicator for live departure displays."""
    if delay_mins > 2:
        return f"({delay_mins}m late)"
    if delay_mins < -2:
        return f"({abs(delay_mins)}m early)"
    return "(Live)"


# ==========================================================================
# GTFS Manager Class
# ==========================================================================


class CarrisGTFSManager:
    """
    Manages GTFS data download, update checking, and SQLite storage.

    Features:
        - HTTP header-based update checking (no full download needed)
        - SQLite conversion with optimized schema (v2.0)
        - PRIMARY KEYs on all tables (composite for stop_times, calendar_dates, shapes)
        - FOREIGN KEYs defined (declarative, not enforced - GTFS data has orphan refs)
        - 13 optimized indexes for common query patterns
        - ANALYZE and VACUUM for query performance
        - Graceful fallback to stale data on errors
    """

    def __init__(self, data_dir: str = CARRIS_DATA_DIR, db_path: str = CARRIS_DB_PATH):
        self.data_dir = data_dir
        self.db_path = db_path
        self.metadata_path = CARRIS_METADATA_PATH

    def _ensure_data_dir(self):
        """Creates data directory if it doesn't exist."""
        os.makedirs(self.data_dir, exist_ok=True)

    def _load_metadata(self) -> Dict[str, Any]:
        """Loads metadata (last update date, etc.)."""
        try:
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")
        return {}

    def _save_metadata(self, data: Dict[str, Any]):
        """Saves metadata to JSON file."""
        try:
            self._ensure_data_dir()
            with open(self.metadata_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save metadata: {e}")

    def _database_has_stops(self) -> bool:
        """Return whether the local Carris SQLite database has stop rows."""
        if not os.path.exists(self.db_path):
            return False

        try:
            with sqlite3.connect(self.db_path) as conn:
                count = conn.execute("SELECT COUNT(*) FROM stops").fetchone()[0]
            return int(count or 0) > 0
        except sqlite3.Error as exc:
            logger.warning(f"Carris database validation failed: {exc}")
            return False

    def restore_database_from_release(self) -> bool:
        """Restore Carris runtime files from the last-known-good release asset."""
        status = ensure_runtime_data_from_release(
            operator_name="Carris Urban",
            target_dir=Path(self.data_dir),
            required_files=("carris.db", "metadata.json"),
            env_prefix=CARRIS_RUNTIME_RELEASE_ENV_PREFIX,
            default_asset=CARRIS_RUNTIME_RELEASE_ASSET,
        )

        if status.ok and status.restored:
            logger.warning(status.message)
        elif not status.ok:
            logger.warning(status.message)

        return status.ok and self._database_has_stops()

    def check_for_updates(self) -> Tuple[bool, Optional[str]]:
        """
        Checks if GTFS data needs updating by examining HTTP headers.

        Uses stream=True to get headers without downloading the full file.
        Extracts date from Content-Disposition filename (e.g., gtfs_2026-01-20.zip).

        Returns:
            Tuple of (needs_update, remote_date)
        """
        try:
            response = requests.get(CARRIS_GTFS_URL, stream=True, timeout=30)
            response.close()

            if response.status_code != 200:
                logger.warning(f"GTFS check failed: HTTP {response.status_code}")
                return False, None

            content_disp = response.headers.get("Content-Disposition", "")
            match = re.search(r"filename=gtfs_(\d{4}-\d{2}-\d{2})\.zip", content_disp)

            if not match:
                logger.warning(f"Could not parse GTFS date from: {content_disp}")
                return True, None

            remote_date = match.group(1)
            metadata = self._load_metadata()
            local_date = metadata.get("gtfs_date")

            if local_date != remote_date:
                logger.info(f"GTFS update available: {local_date} -> {remote_date}")
                return True, remote_date

            logger.info(f"GTFS is up-to-date ({remote_date})")
            return False, remote_date

        except Exception as e:
            logger.debug(f"Carris GTFS update check failed; using cached data when available: {e}")
            return False, None

    def download_gtfs(self) -> Optional[bytes]:
        """Downloads the GTFS ZIP file."""
        try:
            logger.info("Downloading Carris GTFS data (~34MB)...")
            response = requests.get(CARRIS_GTFS_URL, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            size_mb = len(response.content) / (1024 * 1024)
            logger.info(f"Downloaded {size_mb:.2f} MB")
            return response.content

        except Exception as e:
            logger.error(f"Failed to download GTFS: {e}")
            return None

    def convert_to_sqlite(self, gtfs_zip_content: bytes, gtfs_date: str) -> bool:
        """
        Converts GTFS ZIP to SQLite database with optimized schema.

        Schema Features:
            - PRIMARY KEYs on all tables (composite for stop_times, calendar_dates, shapes)
            - FOREIGN KEYs for referential integrity
            - NOT NULL constraints on required GTFS fields
            - Optimized indexes for common query patterns
            - ANALYZE and VACUUM for query optimization
        """
        self._ensure_data_dir()

        if os.path.exists(self.db_path):
            os.remove(self.db_path)

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # .IMPORTANT: Keep FKs OFF during import - real GTFS data often has
            # orphan references (e.g., service_ids in calendar_dates not in calendar,
            # shape_ids in trips not in shapes). FKs are defined for documentation
            # but not enforced to avoid import failures.
            cursor.execute("PRAGMA foreign_keys = OFF")

            # =================================================================
            # Table Definitions with PKs, FKs, and NOT NULL constraints
            # FKs are declarative only (not enforced) for GTFS compatibility
            # =================================================================

            # 1. Agency (no dependencies)
            cursor.execute("""
                CREATE TABLE agency (
                    agency_id TEXT PRIMARY KEY,
                    agency_name TEXT NOT NULL,
                    agency_url TEXT,
                    agency_timezone TEXT NOT NULL,
                    agency_lang TEXT,
                    agency_phone TEXT,
                    agency_fare_url TEXT,
                    agency_email TEXT
                )
            """)

            # 2. Calendar (no dependencies)
            cursor.execute("""
                CREATE TABLE calendar (
                    service_id TEXT PRIMARY KEY,
                    monday INTEGER NOT NULL,
                    tuesday INTEGER NOT NULL,
                    wednesday INTEGER NOT NULL,
                    thursday INTEGER NOT NULL,
                    friday INTEGER NOT NULL,
                    saturday INTEGER NOT NULL,
                    sunday INTEGER NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL
                )
            """)

            # 3. Calendar Dates (depends on calendar)
            cursor.execute("""
                CREATE TABLE calendar_dates (
                    service_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    exception_type INTEGER NOT NULL,
                    PRIMARY KEY (service_id, date),
                    FOREIGN KEY (service_id) REFERENCES calendar(service_id)
                )
            """)

            # 4. Routes (depends on agency)
            cursor.execute("""
                CREATE TABLE routes (
                    route_id TEXT PRIMARY KEY,
                    agency_id TEXT,
                    route_short_name TEXT,
                    route_long_name TEXT,
                    route_desc TEXT,
                    route_type INTEGER NOT NULL,
                    route_url TEXT,
                    route_color TEXT,
                    route_text_color TEXT,
                    FOREIGN KEY (agency_id) REFERENCES agency(agency_id)
                )
            """)

            # 5. Stops (no dependencies)
            cursor.execute("""
                CREATE TABLE stops (
                    stop_id TEXT PRIMARY KEY,
                    stop_code TEXT,
                    stop_name TEXT NOT NULL,
                    stop_desc TEXT,
                    stop_lat REAL NOT NULL,
                    stop_lon REAL NOT NULL,
                    zone_id TEXT,
                    stop_url TEXT,
                    location_type INTEGER,
                    parent_station TEXT,
                    stop_timezone TEXT,
                    wheelchair_boarding INTEGER
                )
            """)

            # 6. Shapes (no dependencies)
            cursor.execute("""
                CREATE TABLE shapes (
                    shape_id TEXT NOT NULL,
                    shape_pt_lat REAL NOT NULL,
                    shape_pt_lon REAL NOT NULL,
                    shape_pt_sequence INTEGER NOT NULL,
                    shape_dist_traveled REAL,
                    PRIMARY KEY (shape_id, shape_pt_sequence)
                )
            """)

            # 7. Trips (depends on routes, calendar, shapes)
            cursor.execute("""
                CREATE TABLE trips (
                    trip_id TEXT PRIMARY KEY,
                    route_id TEXT NOT NULL,
                    service_id TEXT NOT NULL,
                    trip_headsign TEXT,
                    trip_short_name TEXT,
                    direction_id INTEGER,
                    block_id TEXT,
                    shape_id TEXT,
                    wheelchair_accessible INTEGER,
                    bikes_allowed INTEGER,
                    FOREIGN KEY (route_id) REFERENCES routes(route_id),
                    FOREIGN KEY (service_id) REFERENCES calendar(service_id),
                    FOREIGN KEY (shape_id) REFERENCES shapes(shape_id)
                )
            """)

            # 8. Stop Times (depends on trips, stops) - largest table
            cursor.execute("""
                CREATE TABLE stop_times (
                    trip_id TEXT NOT NULL,
                    arrival_time TEXT NOT NULL,
                    departure_time TEXT NOT NULL,
                    stop_id TEXT NOT NULL,
                    stop_sequence INTEGER NOT NULL,
                    stop_headsign TEXT,
                    pickup_type INTEGER,
                    drop_off_type INTEGER,
                    shape_dist_traveled REAL,
                    timepoint INTEGER,
                    PRIMARY KEY (trip_id, stop_sequence),
                    FOREIGN KEY (trip_id) REFERENCES trips(trip_id),
                    FOREIGN KEY (stop_id) REFERENCES stops(stop_id)
                )
            """)

            logger.info("Schema created with PKs and FKs")

            # =================================================================
            # Import Data (order matters for FK constraints)
            # =================================================================

            z = zipfile.ZipFile(BytesIO(gtfs_zip_content))

            # Import order respects foreign key dependencies
            import_order = [
                "agency",
                "calendar",
                "calendar_dates",
                "routes",
                "stops",
                "shapes",
                "trips",
                "stop_times",
            ]

            for table_name in import_order:
                filename = f"{table_name}.txt"

                try:
                    with z.open(filename) as f:
                        content = f.read().decode("utf-8-sig")
                        reader = csv.DictReader(content.splitlines())
                        rows = list(reader)

                        if rows:
                            cols = list(rows[0].keys())
                            placeholders = ",".join(["?" for _ in cols])
                            sql = f"INSERT INTO {table_name} ({','.join(cols)}) VALUES ({placeholders})"

                            data = [
                                [row.get(c, "") or None for c in cols] for row in rows
                            ]
                            cursor.executemany(sql, data)

                    logger.info(f"  {filename}: {len(rows):,} rows")

                except KeyError:
                    if table_name == "shapes":
                        logger.info(f"  {filename}: not found in ZIP (optional)")
                    else:
                        logger.warning(f"  {filename}: not found in ZIP")
                except Exception as e:
                    logger.error(f"  {filename}: error - {e}")

            conn.commit()

            # =================================================================
            # Create Optimized Indexes (after data import for performance)
            # =================================================================

            indexes = [
                # Routes indexes
                "CREATE INDEX idx_routes_short_name ON routes (route_short_name)",
                "CREATE INDEX idx_routes_type ON routes (route_type)",
                # Stops indexes
                "CREATE INDEX idx_stops_name ON stops (stop_name)",
                "CREATE INDEX idx_stops_coords ON stops (stop_lat, stop_lon)",
                # Trips indexes
                "CREATE INDEX idx_trips_route ON trips (route_id)",
                "CREATE INDEX idx_trips_service ON trips (service_id)",
                "CREATE INDEX idx_trips_route_service ON trips (route_id, service_id)",
                # Stop Times indexes (critical for arrivals queries)
                "CREATE INDEX idx_stop_times_stop ON stop_times (stop_id)",
                "CREATE INDEX idx_stop_times_stop_time ON stop_times (stop_id, arrival_time)",
                "CREATE INDEX idx_stop_times_trip_seq ON stop_times (trip_id, stop_sequence)",
                # Calendar indexes
                "CREATE INDEX idx_calendar_dates ON calendar (start_date, end_date)",
                "CREATE INDEX idx_calendar_dates_date ON calendar_dates (date)",
                # Shapes index
                "CREATE INDEX idx_shapes_id ON shapes (shape_id)",
            ]

            logger.info("Creating indexes...")
            for idx_sql in indexes:
                try:
                    cursor.execute(idx_sql)
                except Exception as e:
                    logger.warning(f"Index warning: {e}")

            # =================================================================
            # Optimize Database
            # =================================================================

            logger.info("Running ANALYZE for query optimization...")
            cursor.execute("ANALYZE")

            conn.commit()
            conn.close()

            # VACUUM must be outside transaction (separate connection)
            logger.info("Running VACUUM to reclaim space...")
            conn = sqlite3.connect(self.db_path)
            conn.execute("VACUUM")
            conn.close()

            self._save_metadata(
                {
                    "gtfs_date": gtfs_date,
                    "updated_at": datetime.now().isoformat(),
                    "db_path": os.path.basename(self.db_path),
                    "schema_version": "2.0",  # Indicates optimized schema
                    "features": [
                        "primary_keys",
                        "foreign_keys_declarative",
                        "indexes",
                        "analyzed",
                    ],
                }
            )

            db_size = os.path.getsize(self.db_path) / (1024 * 1024)
            logger.info(
                f"SQLite database created: {db_size:.1f} MB (optimized schema v2.0)"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to convert GTFS to SQLite: {e}")
            return False

    def ensure_database(self, force_update: bool = False) -> bool:
        """Ensures database exists and is up-to-date."""
        if not force_update and os.path.exists(self.db_path):
            needs_update, remote_date = self.check_for_updates()
            if not needs_update:
                return True
        else:
            needs_update = True
            _, remote_date = self.check_for_updates()

        if needs_update or force_update:
            gtfs_content = self.download_gtfs()
            if gtfs_content:
                return self.convert_to_sqlite(gtfs_content, remote_date or "unknown")
            else:
                if os.path.exists(self.db_path):
                    logger.warning("Using stale database due to download failure")
                    return True
                if self.restore_database_from_release():
                    logger.warning("Using Carris last-known-good release backup due to download failure")
                    return True
                return False

        return True


# ==========================================================================
# Database Query Helpers
# ==========================================================================


def _get_db_connection() -> Optional[sqlite3.Connection]:
    """Gets a database connection, ensuring DB exists."""
    manager = CarrisGTFSManager()
    if not manager.ensure_database():
        return None

    try:
        conn = sqlite3.connect(CARRIS_DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return None


def _get_active_services(
    conn: sqlite3.Connection, date: Optional[datetime] = None
) -> List[str]:
    """Gets active service_ids for a given date."""
    if date is None:
        date = datetime.now()

    date_str = date.strftime("%Y%m%d")
    day_of_week = date.strftime("%A").lower()

    cursor = conn.cursor()

    cursor.execute(
        f"""
        SELECT service_id FROM calendar
        WHERE {day_of_week} = 1
        AND start_date <= ? AND end_date >= ?
    """,
        (date_str, date_str),
    )

    active_services = set(row["service_id"] for row in cursor.fetchall())

    cursor.execute(
        """
        SELECT service_id FROM calendar_dates
        WHERE date = ? AND exception_type = 1
    """,
        (date_str,),
    )
    for row in cursor.fetchall():
        active_services.add(row["service_id"])

    cursor.execute(
        """
        SELECT service_id FROM calendar_dates
        WHERE date = ? AND exception_type = 2
    """,
        (date_str,),
    )
    for row in cursor.fetchall():
        active_services.discard(row["service_id"])

    return list(active_services)


def _get_route_info_by_id(conn: sqlite3.Connection, route_id: str) -> Optional[Dict]:
    """Gets route information from route_id."""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT route_id, route_short_name, route_long_name, route_type
        FROM routes WHERE route_id = ?
    """,
        (route_id,),
    )
    row = cursor.fetchone()
    return dict(row) if row else None


def _get_stop_info_by_id(conn: sqlite3.Connection, stop_id: str) -> Optional[Dict]:
    """Gets stop information from stop_id."""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT stop_id, stop_name, stop_lat, stop_lon, stop_code
        FROM stops WHERE stop_id = ?
    """,
        (stop_id,),
    )
    row = cursor.fetchone()
    return dict(row) if row else None


def _get_trip_info_by_id(conn: sqlite3.Connection, trip_id: str) -> Optional[Dict]:
    """Gets trip information including route details."""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT t.trip_id, t.trip_headsign, t.direction_id, t.route_id,
               r.route_short_name, r.route_long_name
        FROM trips t
        JOIN routes r ON t.route_id = r.route_id
        WHERE t.trip_id = ?
    """,
        (trip_id,),
    )
    row = cursor.fetchone()
    return dict(row) if row else None


# ==========================================================================
# GTFS-RT Real-Time Data
# ==========================================================================


def fetch_gtfs_rt_vehicles(use_cache: bool = True) -> List[Dict[str, Any]]:
    """
    Fetches real-time vehicle positions from Carris GTFS-RT feed.

    Implements retry logic with exponential backoff because the Carris
    GTFS-RT endpoint can be slow or intermittently unavailable.

    Args:
        use_cache: Whether to use cached data if available (default: True)

    Returns:
        List of vehicle dictionaries with parsed GTFS-RT data.
    """
    if not GTFS_RT_AVAILABLE:
        logger.error("gtfs-realtime-bindings library not installed")
        return []

    now = time.time()
    if use_cache and _gtfs_rt_cache["data"] is not None:
        age = now - _gtfs_rt_cache["timestamp"]
        if age < _gtfs_rt_cache["ttl"]:
            logger.debug(f"Using cached GTFS-RT data (age: {age:.1f}s)")
            generated_at = None
            if _gtfs_rt_feed_meta.get("generated_at"):
                try:
                    generated_at = datetime.fromisoformat(_gtfs_rt_feed_meta["generated_at"])
                except ValueError:
                    generated_at = None
            _update_gtfs_rt_feed_meta(
                source="cache",
                generated_at=generated_at,
                data_age_seconds=age,
                last_error=None,
                vehicle_count=len(_gtfs_rt_cache["data"] or []),
            )
            return _gtfs_rt_cache["data"]

    # Retry loop with exponential backoff
    last_error = None
    for attempt in range(1, REALTIME_MAX_RETRIES + 1):
        try:
            logger.info(
                f"GTFS-RT fetch attempt {attempt}/{REALTIME_MAX_RETRIES}..."
            )
            response = requests.get(CARRIS_GTFS_RT_URL, timeout=REALTIME_TIMEOUT)
            response.raise_for_status()

            feed = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(response.content)

            vehicles = []
            feed_timestamp = feed.header.timestamp
            generated_at = (
                datetime.fromtimestamp(feed_timestamp)
                if feed_timestamp
                else datetime.now()
            )

            for entity in feed.entity:
                if not entity.HasField("vehicle"):
                    continue

                v = entity.vehicle
                vehicle_data = {
                    "entity_id": entity.id,
                    "feed_timestamp": feed_timestamp,
                    "trip_id": v.trip.trip_id if v.HasField("trip") else None,
                    "route_id": v.trip.route_id if v.HasField("trip") else None,
                    "direction_id": v.trip.direction_id
                    if v.HasField("trip") and v.trip.HasField("direction_id")
                    else None,
                    "latitude": v.position.latitude
                    if v.HasField("position")
                    else None,
                    "longitude": v.position.longitude
                    if v.HasField("position")
                    else None,
                    "bearing": v.position.bearing
                    if v.HasField("position") and v.position.HasField("bearing")
                    else None,
                    "speed": v.position.speed
                    if v.HasField("position") and v.position.HasField("speed")
                    else None,
                    "vehicle_id": v.vehicle.id if v.HasField("vehicle") else None,
                    "vehicle_label": v.vehicle.label
                    if v.HasField("vehicle") and v.vehicle.label
                    else None,
                    "license_plate": v.vehicle.license_plate
                    if v.HasField("vehicle") and v.vehicle.license_plate
                    else None,
                    "current_status": VEHICLE_STATUS.get(
                        v.current_status, f"UNKNOWN({v.current_status})"
                    ),
                    "current_status_code": v.current_status,
                    "stop_id": v.stop_id if v.stop_id else None,
                    "timestamp": v.timestamp,
                }
                vehicles.append(vehicle_data)

            logger.info(
                f"Fetched {len(vehicles)} vehicles from GTFS-RT feed "
                f"(attempt {attempt})"
            )

            _gtfs_rt_cache["data"] = vehicles
            _gtfs_rt_cache["timestamp"] = time.time()
            _update_gtfs_rt_feed_meta(
                source="live",
                generated_at=generated_at,
                data_age_seconds=max(0.0, time.time() - feed_timestamp) if feed_timestamp else 0.0,
                last_error=None,
                vehicle_count=len(vehicles),
            )

            return vehicles

        except requests.exceptions.Timeout:
            last_error = "timeout"
            logger.warning(
                f"GTFS-RT request timed out (attempt {attempt}/{REALTIME_MAX_RETRIES})"
            )
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            logger.warning(
                f"GTFS-RT request failed (attempt {attempt}/{REALTIME_MAX_RETRIES}): {e}"
            )
        except Exception as e:
            last_error = str(e)
            logger.error(f"Error parsing GTFS-RT data: {e}")
            # Parse errors are not retryable
            break

        # Wait before retrying (exponential backoff)
        if attempt < REALTIME_MAX_RETRIES:
            wait_time = REALTIME_RETRY_BACKOFF * attempt
            logger.info(f"Retrying in {wait_time}s...")
            time.sleep(wait_time)

    # All retries exhausted
    logger.error(
        f"GTFS-RT: All {REALTIME_MAX_RETRIES} attempts failed. "
        f"Last error: {last_error}"
    )
    if _gtfs_rt_cache["data"]:
        cache_age = now - _gtfs_rt_cache["timestamp"]
        logger.info(f"Using cached GTFS-RT data (age: {cache_age:.0f}s)")
        generated_at = None
        if _gtfs_rt_feed_meta.get("generated_at"):
            try:
                generated_at = datetime.fromisoformat(_gtfs_rt_feed_meta["generated_at"])
            except ValueError:
                generated_at = None
        _update_gtfs_rt_feed_meta(
            source="stale_cache",
            generated_at=generated_at,
            data_age_seconds=cache_age,
            last_error=last_error,
            vehicle_count=len(_gtfs_rt_cache["data"] or []),
        )
        return _gtfs_rt_cache["data"]
    _update_gtfs_rt_feed_meta(
        source="unavailable",
        generated_at=None,
        data_age_seconds=None,
        last_error=last_error,
        vehicle_count=0,
    )
    return []


def enrich_vehicle_with_static_data(vehicle: Dict, conn: sqlite3.Connection) -> Dict:
    """Enriches a GTFS-RT vehicle with static GTFS data."""
    enriched = vehicle.copy()

    # Get route info (short name, long name)
    if vehicle.get("route_id"):
        route_info = _get_route_info_by_id(conn, vehicle["route_id"])
        if route_info:
            enriched["route_short_name"] = route_info["route_short_name"]
            enriched["route_long_name"] = _clean_carris_headsign(route_info["route_long_name"])
            enriched["is_tram"] = route_info["route_short_name"].endswith("E")

    # Get stop name
    if vehicle.get("stop_id"):
        stop_info = _get_stop_info_by_id(conn, vehicle["stop_id"])
        if stop_info:
            enriched["stop_name"] = stop_info["stop_name"]

    # Get trip headsign (destination)
    if vehicle.get("trip_id"):
        trip_info = _get_trip_info_by_id(conn, vehicle["trip_id"])
        if trip_info:
            # Use trip_headsign if available, otherwise fall back to route_long_name
            enriched["trip_headsign"] = _resolve_carris_headsign(
                trip_info.get("trip_headsign"),
                trip_info.get("route_long_name"),
                trip_info.get("direction_id"),
            )

    # Final fallback: if trip_headsign is still None, use route_long_name
    if not enriched.get("trip_headsign") and enriched.get("route_long_name"):
        enriched["trip_headsign"] = _clean_carris_headsign(enriched["route_long_name"])

    return enriched


def get_vehicles_for_route(route_short_name: str) -> List[Dict]:
    """Gets all real-time vehicles for a specific route."""
    vehicles = fetch_gtfs_rt_vehicles()
    conn = _get_db_connection()

    if not conn:
        return []

    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT route_id FROM routes WHERE route_short_name = ?",
            (route_short_name,),
        )

        route_ids = [row["route_id"] for row in cursor.fetchall()]

        result = []
        for v in vehicles:
            if v.get("route_id") in route_ids:
                result.append(enrich_vehicle_with_static_data(v, conn))

        conn.close()
        return result

    except Exception as e:
        logger.error(f"Error getting vehicles for route {route_short_name}: {e}")
        conn.close()
        return []


def get_vehicle_eta_at_stop(
    vehicle: Dict, target_stop_id: str, conn: sqlite3.Connection
) -> Optional[Dict]:
    """Estimates arrival time of a vehicle at a target stop."""
    trip_id = vehicle.get("trip_id")
    current_stop_id = vehicle.get("stop_id")

    if not trip_id or not current_stop_id:
        return None

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT stop_sequence, departure_time
        FROM stop_times
        WHERE trip_id = ? AND stop_id = ?
    """,
        (trip_id, current_stop_id),
    )

    current_row = cursor.fetchone()
    if not current_row:
        return None

    current_seq = current_row["stop_sequence"]
    current_scheduled_dep = current_row["departure_time"]

    cursor.execute(
        """
        SELECT stop_sequence, arrival_time, departure_time
        FROM stop_times
        WHERE trip_id = ? AND stop_id = ?
    """,
        (trip_id, target_stop_id),
    )

    target_row = cursor.fetchone()
    if not target_row:
        return None

    target_seq = target_row["stop_sequence"]
    target_scheduled_arr = target_row["arrival_time"]

    if current_seq >= target_seq:
        return None

    current_mins = time_str_to_minutes(current_scheduled_dep)
    target_mins = time_str_to_minutes(target_scheduled_arr)
    scheduled_travel_mins = target_mins - current_mins

    now = datetime.now()
    now_mins = now.hour * 60 + now.minute
    scheduled_now_mins = time_str_to_minutes(current_scheduled_dep)
    delay_mins = now_mins - scheduled_now_mins

    estimated_arrival_mins = target_mins + max(0, delay_mins)

    return {
        "target_stop_id": target_stop_id,
        "scheduled_arrival": target_scheduled_arr[:5],
        "estimated_arrival": minutes_to_time_str(estimated_arrival_mins),
        "scheduled_travel_mins": scheduled_travel_mins,
        "current_delay_mins": delay_mins,
        "stops_remaining": target_seq - current_seq,
    }


def get_next_arrivals_at_stop(stop_id: str, limit: int = 10) -> List[Dict]:
    """Gets next scheduled and real-time arrivals at a stop."""
    conn = _get_db_connection()
    if not conn:
        return []

    try:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")

        active_services = _get_active_services(conn, now)

        if not active_services:
            logger.warning("No active services found for today")
            conn.close()
            return []

        cursor = conn.cursor()
        placeholders = ",".join(["?" for _ in active_services])

        cursor.execute(
            f"""
            SELECT st.trip_id, st.arrival_time, st.departure_time, st.stop_sequence,
                   t.trip_headsign, t.route_id, t.direction_id,
                   r.route_short_name, r.route_long_name
            FROM stop_times st
            JOIN trips t ON st.trip_id = t.trip_id
            JOIN routes r ON t.route_id = r.route_id
            WHERE st.stop_id = ?
            AND t.service_id IN ({placeholders})
            AND st.departure_time >= ?
            ORDER BY st.departure_time
            LIMIT ?
        """,
            (stop_id, *active_services, current_time, limit * 2),
        )

        scheduled = cursor.fetchall()

        rt_vehicles = fetch_gtfs_rt_vehicles()
        rt_by_trip = {v["trip_id"]: v for v in rt_vehicles if v.get("trip_id")}

        arrivals = []
        for row in scheduled:
            trip_id = row["trip_id"]
            scheduled_time = row["departure_time"][:5]

            arrival = {
                "trip_id": trip_id,
                "route_short_name": row["route_short_name"],
                "route_long_name": row["route_long_name"],
                "headsign": row["trip_headsign"] or row["route_long_name"],
                "scheduled_time": scheduled_time,
                "estimated_time": scheduled_time,
                "is_realtime": False,
                "delay_mins": 0,
                "vehicle_id": None,
                "is_tram": row["route_short_name"].endswith("E"),
            }

            if trip_id in rt_by_trip:
                vehicle = rt_by_trip[trip_id]
                arrival["is_realtime"] = True
                arrival["vehicle_id"] = vehicle.get("vehicle_id")
                arrival["license_plate"] = vehicle.get("license_plate")

                eta_info = get_vehicle_eta_at_stop(vehicle, stop_id, conn)
                if eta_info:
                    arrival["estimated_time"] = eta_info["estimated_arrival"]
                    arrival["delay_mins"] = eta_info["current_delay_mins"]
                    arrival["stops_remaining"] = eta_info["stops_remaining"]

            arrivals.append(arrival)

        arrivals.sort(key=lambda x: time_str_to_minutes(x["estimated_time"]))

        conn.close()
        return arrivals[:limit]

    except Exception as e:
        logger.error(f"Error getting arrivals at stop {stop_id}: {e}")
        conn.close()
        return []


def _get_directional_departures_for_route(
    conn: sqlite3.Connection,
    origin_stops: List[Dict[str, Any]],
    dest_stops: List[Dict[str, Any]],
    route_short_name: str,
    active_services: List[str],
    current_time: str,
    limit: int = 4,
) -> List[Dict[str, Any]]:
    """Returns next direction-safe departures for a route from origin toward destination."""
    if not active_services or not origin_stops or not dest_stops:
        return []

    cursor = conn.cursor()
    rt_vehicles = fetch_gtfs_rt_vehicles()
    rt_by_trip = {v["trip_id"]: v for v in rt_vehicles if v.get("trip_id")}

    origin_ids = [s["id"] for s in origin_stops[:12]]
    dest_ids = [s["id"] for s in dest_stops[:12]]
    ph_o = ",".join(["?" for _ in origin_ids])
    ph_d = ",".join(["?" for _ in dest_ids])
    ph_s = ",".join(["?" for _ in active_services])

    cursor.execute(
        f"""
        SELECT st.trip_id,
               st.stop_id AS origin_stop_id,
               s.stop_name AS origin_stop_name,
               st.departure_time,
               t.trip_headsign,
               t.direction_id,
               r.route_long_name,
               MIN(st_d.arrival_time) AS target_arrival_time
        FROM stop_times st
        JOIN trips t ON st.trip_id = t.trip_id
        JOIN routes r ON t.route_id = r.route_id
        JOIN stops s ON st.stop_id = s.stop_id
        JOIN stop_times st_d ON st.trip_id = st_d.trip_id
            AND st_d.stop_id IN ({ph_d})
            AND st.stop_sequence < st_d.stop_sequence
        WHERE st.stop_id IN ({ph_o})
          AND r.route_short_name = ?
          AND t.service_id IN ({ph_s})
          AND st.departure_time >= ?
        GROUP BY st.trip_id, st.stop_id, s.stop_name, st.departure_time, t.trip_headsign, t.direction_id, r.route_long_name
        ORDER BY st.departure_time
        LIMIT ?
        """,
        dest_ids + origin_ids + [route_short_name] + active_services + [current_time, limit],
    )

    departures: List[Dict[str, Any]] = []
    for row in cursor.fetchall():
        headsign = _resolve_carris_headsign(
            row["trip_headsign"],
            row["route_long_name"],
            row["direction_id"],
        )
        scheduled_departure = row["departure_time"][:5]
        estimated_departure = scheduled_departure
        delay_mins = 0
        is_realtime = False

        vehicle = rt_by_trip.get(row["trip_id"])
        if vehicle:
            eta_info = get_vehicle_eta_at_stop(vehicle, row["origin_stop_id"], conn)
            if eta_info:
                estimated_departure = eta_info["estimated_arrival"]
                delay_mins = eta_info["current_delay_mins"]
                is_realtime = True

        travel_mins = None
        if row["target_arrival_time"]:
            dep_mins = time_str_to_minutes(row["departure_time"])
            arr_mins = time_str_to_minutes(row["target_arrival_time"])
            diff = arr_mins - dep_mins
            if diff < 0:
                diff += 24 * 60
            if diff > 0:
                travel_mins = diff

        departures.append(
            {
                "trip_id": row["trip_id"],
                "origin_stop_id": row["origin_stop_id"],
                "origin_stop_name": row["origin_stop_name"],
                "headsign": headsign,
                "scheduled_departure": scheduled_departure,
                "estimated_departure": estimated_departure,
                "delay_mins": delay_mins,
                "is_realtime": is_realtime,
                "travel_mins": travel_mins,
            }
        )

    departures.sort(key=lambda item: time_str_to_minutes(item["estimated_departure"]))
    return departures


# ==========================================================================
# Tool Functions
# ==========================================================================


@tool
def carris_get_stops(query: str = "", limit: Optional[int] = None) -> str:
    """
    Searches Carris (Lisbon urban) bus and tram stops.

    Args:
        query: Search term for stop name (empty = list all stops)
        limit: Maximum results (optional). If not specified:
               - Generic listing: 50 results
               - Search query: Unlimited results (shows all matches)

    Returns:
        Formatted list of matching stops with ID, name, and coordinates.
    """
    conn = _get_db_connection()
    if not conn:
        return "Base de dados da Carris indisponível."

    try:
        cursor = conn.cursor()

        # Determine effective limit
        if limit is None:
            effective_limit = 1000 if query else 50
        else:
            effective_limit = limit

        if query:
            rows = _search_stop_rows(conn, query, effective_limit)
        else:
            sql = """
                SELECT stop_id, stop_name, stop_lat, stop_lon, stop_code
                FROM stops ORDER BY stop_name LIMIT ?
            """
            cursor.execute(sql, (effective_limit,))
            rows = cursor.fetchall()
        conn.close()

        if not rows:
            return f"Nenhuma paragem Carris encontrada para '{query}'"

        shown_rows = rows[:5]
        response = f"### 🚌 **Carris stops near '{query}'** ({len(shown_rows)} shown)\n\n"

        for row in shown_rows:
            response += f"**🚏 {row['stop_name']}**\n"
            response += f"    - 🆔 **Stop ID:** `{row['stop_id']}`\n"
            if row["stop_code"]:
                response += f"    - 🔢 **Stop code:** `{row['stop_code']}`\n"
            if row["stop_lat"] and row["stop_lon"]:
                response += (
                    "    - 📍 "
                    f"[Open map](https://www.google.com/maps/search/?api=1&query={row['stop_lat']:.5f}%2C{row['stop_lon']:.5f})\n"
                )
            response += "\n"

        if len(rows) > len(shown_rows):
            response += f"... and {len(rows) - len(shown_rows)} more stops.\n"
        return response

    except Exception as e:
        logger.error(f"Error searching stops: {e}")
        return f"Erro ao pesquisar paragens: {e}"


@tool
def carris_get_routes(route_type: str = "", route_id: str = "", limit: int = 50) -> str:
    """
    Gets Carris (Lisbon urban) bus and tram routes.

    Args:
        route_type: Filter by type - "bus", "tram", "elétrico", "autocarro" (optional)
        route_id: Search for specific route ID or short name (optional)
        limit: Maximum results (default: 50)

    Returns:
        Formatted list of routes with ID, name, and type.
    """
    conn = _get_db_connection()
    if not conn:
        return "Base de dados da Carris indisponível."

    try:
        cursor = conn.cursor()

        conditions = []
        params = []

        if route_type:
            type_lower = route_type.lower().strip()
            if type_lower in ["tram", "elétrico", "eletrico"]:
                conditions.append("route_short_name LIKE '%E'")
            elif type_lower in ["bus", "autocarro"]:
                conditions.append("route_short_name NOT LIKE '%E'")

        if route_id:
            conditions.append("(route_id LIKE ? OR route_short_name LIKE ?)")
            params.extend([f"%{route_id}%", f"%{route_id}%"])

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        sql = f"""
            SELECT route_id, route_short_name, route_long_name, route_type
            FROM routes WHERE {where_clause} ORDER BY route_short_name LIMIT ?
        """
        params.append(limit)

        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            filter_msg = f" (tipo: {route_type})" if route_type else ""
            return f"Nenhuma rota Carris encontrada{filter_msg}"

        if route_id:
            variants = []
            route_short_names = []
            for route in rows:
                short_name = route["route_short_name"]
                long_name = route["route_long_name"] or "Sem designação na fonte"
                if short_name not in route_short_names:
                    route_short_names.append(short_name)
                if long_name not in variants:
                    variants.append(long_name)

            title_route = ", ".join(route_short_names) or route_id
            icon = "🚋" if any(name.endswith("E") for name in route_short_names) else "🚌"
            response = f"### {icon} **Carris Urban route {title_route}**\n\n"
            response += "- **Operator:** Carris Urban\n"
            response += "- **Route variants in GTFS:**\n"
            for variant in variants[:6]:
                response += f"    - {variant}\n"
            if len(variants) > 6:
                response += f"    - ... and {len(variants) - 6} more variants\n"
            return response.strip()

        trams = [r for r in rows if r["route_short_name"].endswith("E")]
        buses = [r for r in rows if not r["route_short_name"].endswith("E")]

        response = "Rotas Carris (Lisboa Urbano)\n"
        response += "=" * 45 + "\n\n"

        if trams:
            response += "ELÉTRICOS\n" + "-" * 30 + "\n"
            for r in trams:
                response += (
                    f"   {r['route_short_name']}: {r['route_long_name'] or 'N/A'}\n"
                )
            response += "\n"

        if buses:
            response += "AUTOCARROS\n" + "-" * 30 + "\n"
            for r in buses[:25]:
                response += (
                    f"   {r['route_short_name']}: {r['route_long_name'] or 'N/A'}\n"
                )
            if len(buses) > 25:
                response += f"   ... e mais {len(buses) - 25} rotas de autocarro\n"
            response += "\n"

        response += f"Total: {len(trams)} elétricos + {len(buses)} autocarros\n"
        return response

    except Exception as e:
        logger.error(f"Error getting routes: {e}")
        return f"Erro ao obter rotas: {e}"


@tool
def carris_get_arrivals(stop_id: str, limit: int = 10) -> str:
    """
    Gets real-time arrivals at a Carris stop combining schedule and live tracking.

    This is the PRIMARY tool for "when is the next bus/tram at stop X".
    Combines GTFS schedule with GTFS-RT live vehicle positions.

    Args:
        stop_id: Stop ID (from carris_get_stops)
        limit: Maximum arrivals to show (default: 10)

    Returns:
        Formatted list of upcoming arrivals with times and delay info.
    """
    conn = _get_db_connection()
    if not conn:
        return "Base de dados da Carris indisponível."

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT stop_name FROM stops WHERE stop_id = ?", (stop_id,))
        stop_row = cursor.fetchone()

        if not stop_row:
            conn.close()
            return f"Paragem '{stop_id}' não encontrada."

        stop_name = stop_row["stop_name"]
        conn.close()

        arrivals = get_next_arrivals_at_stop(stop_id, limit)

        if not arrivals:
            return f"Sem mais partidas hoje na paragem {stop_name} (ID: {stop_id})."

        response = f"Próximas Chegadas: {stop_name}\n"
        response += (
            f"   ID: {stop_id} | Atualizado: {datetime.now().strftime('%H:%M')}\n"
        )
        response += "=" * 55 + "\n\n"
        freshness_note = _build_gtfs_rt_freshness_note()
        if freshness_note:
            response += freshness_note + "\n\n"

        for arr in arrivals:
            vehicle_type = "Elétrico" if arr["is_tram"] else "Autocarro"
            rt_indicator = "[REAL-TIME]" if arr["is_realtime"] else "[SCHEDULE]"

            if arr["is_realtime"] and arr["delay_mins"] != 0:
                if arr["delay_mins"] > 0:
                    time_display = (
                        f"{arr['estimated_time']} ({arr['delay_mins']} min late)"
                    )
                else:
                    time_display = (
                        f"{arr['estimated_time']} ({abs(arr['delay_mins'])} min early)"
                    )
            else:
                time_display = arr["scheduled_time"]

            response += f"{rt_indicator} {vehicle_type} {arr['route_short_name']} -> {arr['headsign'][:35]}\n"
            response += f"   Hora: {time_display}\n"

            if arr["is_realtime"] and arr.get("stops_remaining"):
                response += f"   {arr['stops_remaining']} stops remaining\n"

            if arr.get("vehicle_id"):
                plate_info = (
                    f" | Plate: {arr['license_plate']}"
                    if arr.get("license_plate")
                    else ""
                )
                response += f"   Vehicle: {arr['vehicle_id']}{plate_info}\n"

            response += "\n"

        response += "-" * 55 + "\n"
        response += "[REAL-TIME] = Vehicle GPS data | [SCHEDULE] = Scheduled time\n"
        return response

    except Exception as e:
        logger.error(f"Error getting arrivals: {e}")
        return f"Error getting arrivals: {e}"


@tool
def carris_get_next_departures(
    stop_id: str, start_time: str = "", route_short_name: str = "", limit: int = 25
) -> str:
    """
    Gets upcoming scheduled departures for a Carris stop, ENRICHED with Real-Time data.

    Args:
        stop_id: Stop ID (from carris_get_stops)
        start_time: Optional time (HH:MM) to see schedule for a specific time (default: now)
        route_short_name: Optional filter by line number (e.g. "736", "15E")
        limit: Maximum departures (default: 25)

    Returns:
        Formatted schedule with times, route info, and real-time updates.
    """
    conn = _get_db_connection()
    if not conn:
        return "Carris database unavailable."

    try:
        cursor = conn.cursor()

        original_stop_input = str(stop_id or "").strip()
        resolved_from_name = False

        cursor.execute("SELECT stop_name FROM stops WHERE stop_id = ?", (original_stop_input,))
        stop_row = cursor.fetchone()

        if not stop_row:
            name_matches = _search_stop_rows(conn, original_stop_input, limit=1)
            if not name_matches:
                conn.close()
                return (
                    f"Stop '{original_stop_input}' not found. "
                    "Use carris_get_stops first and pass one returned Stop ID."
                )
            matched_stop = name_matches[0]
            stop_id = str(matched_stop["stop_id"])
            stop_row = matched_stop
            resolved_from_name = True
        else:
            stop_id = original_stop_input

        stop_name = stop_row["stop_name"]

        # Time handling
        is_future_query = False
        if start_time:
            try:
                datetime.strptime(start_time, "%H:%M")
                current_time = f"{start_time}:00"
                # If user asks for a specific time, assume they want static schedule or it's a future plan
                # We'll still try to match RT if the time is close to now, but flagged as future query if distinct
                now_str = datetime.now().strftime("%H:%M")
                if (
                    start_time[:2] != now_str[:2]
                ):  # Simple heuristic to detect future hours
                    is_future_query = True
            except ValueError:
                conn.close()
                return "Invalid time format. Use HH:MM."
        else:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")

        active_services = _get_active_services(conn, datetime.now())

        if not active_services:
            conn.close()
            return "No active services found for today."

        # Hub logic (same stop name = multiple stop_ids)
        cursor.execute("SELECT stop_id FROM stops WHERE stop_name = ?", (stop_name,))
        stop_ids = [row["stop_id"] for row in cursor.fetchall()]

        placeholders = ",".join(["?" for _ in active_services])
        stop_placeholders = ",".join(["?" for _ in stop_ids])

        params = list(stop_ids) + list(active_services) + [current_time]

        route_filter = ""
        if route_short_name:
            route_filter = "AND r.route_short_name = ?"
            params.append(route_short_name)

        # Added st.trip_id to query to enable real-time matching
        # Terminus filter: exclude last stop of each trip (buses terminating
        # here are arrivals, not departures)
        sql = f"""
            SELECT st.trip_id, st.departure_time, r.route_short_name, r.route_long_name, t.trip_headsign, t.direction_id
            FROM stop_times st
            JOIN trips t ON st.trip_id = t.trip_id
            JOIN routes r ON t.route_id = r.route_id
            WHERE st.stop_id IN ({stop_placeholders})
            AND t.service_id IN ({placeholders})
            AND st.departure_time >= ?
            AND st.stop_sequence < (
                SELECT MAX(st2.stop_sequence)
                FROM stop_times st2
                WHERE st2.trip_id = st.trip_id
            )
            {route_filter}
            ORDER BY st.departure_time LIMIT ?
        """

        params.append(limit)
        cursor.execute(sql, params)
        rows = cursor.fetchall()

        if not rows:
            conn.close()
            return f"No more departures found after {current_time[:5]}."

        # Fetch Real-Time Data if not a distant future query
        rt_by_trip = {}
        if not is_future_query:
            try:
                rt_vehicles = fetch_gtfs_rt_vehicles()
                rt_by_trip = {v["trip_id"]: v for v in rt_vehicles if v.get("trip_id")}
            except Exception as e:
                logger.error(f"Failed to fetch RT data in next_departures: {e}")

        # Group by (Route, Destination)
        # key: (route_short_name, destination) -> list of formatted time strings
        departures = {}

        for row in rows:
            trip_id = row["trip_id"]
            route_short = row["route_short_name"]
            dest = row["trip_headsign"]

            # Destination fallback logic
            if not dest or str(dest).lower() == "none":
                long_name = row["route_long_name"] or ""
                parts = long_name.split(" - ")
                if len(parts) == 2:
                    if row["direction_id"] == 0:
                        dest = parts[1]
                    else:
                        dest = parts[0]
                else:
                    dest = long_name or "Unknown"

            group_key = (route_short, dest)
            if group_key not in departures:
                departures[group_key] = []

            # Calculate Display Time
            scheduled_time = row["departure_time"][:5]
            display_time = scheduled_time

            if trip_id in rt_by_trip:
                vehicle = rt_by_trip[trip_id]
                # Use helper to calculate exact ETA at THIS stop considering delays
                eta_info = get_vehicle_eta_at_stop(vehicle, stop_id, conn)
                if eta_info:
                    estimated_time = eta_info["estimated_arrival"]
                    delay = eta_info["current_delay_mins"]

                    # Formatting: **17:45** (Live)
                    # We use simple string indicators
                    if delay > 2:
                        display_time = f"**{estimated_time}** ({delay}m late)"
                    elif delay < -2:
                        display_time = f"**{estimated_time}** ({abs(delay)}m early)"
                    else:
                        display_time = f"**{estimated_time}** (Live)"

            departures[group_key].append(display_time)

        conn.close()

        # Check if we actually used any real-time data in the final list
        has_realtime_data = any(
            any(t.startswith("**") for t in times) for times in departures.values()
        )
        response = f"🚌 **Next Departures from {stop_name}**  \n"
        if resolved_from_name:
            response += f"   Resolved '{original_stop_input}' to Stop ID `{stop_id}`.  \n"
        if has_realtime_data:
            response += "   (📡 Real-Time Data Active)  \n"
        freshness_note = _build_gtfs_rt_freshness_note() if not is_future_query else ""
        if freshness_note:
            response += f"   {freshness_note}  \n"
        response += "---  \n"

        for (route, dest), times in departures.items():
            # Show top 5 times per destination
            shown_times = times[:5]
            remaining = len(times) - 5

            times_str = ", ".join(shown_times)
            if remaining > 0:
                times_str += f" (+{remaining} more)"

            # USER REQUEST: "DIZER O NOME DA ROTA"
            response += f"📍 **[{route}] Para {dest}**  \n"
            response += f"   🕒 {times_str}  \n\n"

        total_shown = sum(len(times) for times in departures.values())
        response += f"Showing next {total_shown} departures.  \n"
        return response

    except Exception as e:
        if conn:
            conn.close()
        logger.error(f"Error getting schedule: {e}")
        return f"Error getting schedule: {e}"


@tool
def carris_find_routes_between(
    origin: str, destination: str, search_radius_km: float = 0.4
) -> str:
    """
    Finds Carris routes connecting two locations with real-time estimates.

    Args:
        origin: Origin location (e.g., "Rossio", "Praça do Comércio")
        destination: Destination (e.g., "Belém", "Parque das Nações")
        search_radius_km: Initial search radius (default: 0.4km)

    Returns:
        Route options with real-time waiting times.
    """
    conn = _get_db_connection()
    if not conn:
        return "Carris database unavailable."

    try:
        cursor = conn.cursor()

        response = f"Routes: {origin} -> {destination}\n"
        response += "=" * 55 + "\n\n"
        ambiguity_note = build_location_ambiguity_preamble(origin, destination, language="pt")
        if ambiguity_note:
            conn.close()
            return ambiguity_note

        origin_lat, origin_lon, origin_name = geocode_location(origin)
        dest_lat, dest_lon, dest_name = geocode_location(destination)

        if origin_lat is None:
            fallback_origin_rows = _search_stop_rows(conn, origin, limit=1)
            row = fallback_origin_rows[0] if fallback_origin_rows else None
            if row:
                origin_lat, origin_lon, origin_name = (
                    row["stop_lat"],
                    row["stop_lon"],
                    row["stop_name"],
                )
            else:
                conn.close()
                return f"Could not locate '{origin}'."

        if dest_lat is None:
            fallback_dest_rows = _search_stop_rows(conn, destination, limit=1)
            row = fallback_dest_rows[0] if fallback_dest_rows else None
            if row:
                dest_lat, dest_lon, dest_name = (
                    row["stop_lat"],
                    row["stop_lon"],
                    row["stop_name"],
                )
            else:
                conn.close()
                return f"Could not locate '{destination}'."

        # Ensure coordinates are floats before passing to find_stops_near
        origin_lat = float(origin_lat) if origin_lat is not None else 0.0
        origin_lon = float(origin_lon) if origin_lon is not None else 0.0
        dest_lat = float(dest_lat) if dest_lat is not None else 0.0
        dest_lon = float(dest_lon) if dest_lon is not None else 0.0

        origin_display = get_location_display_name(origin) or (
            origin_name.split(",")[0] if origin_name else origin
        )
        dest_display = get_location_display_name(destination) or (
            dest_name.split(",")[0] if dest_name else destination
        )

        response += f"   From: {origin_display}\n"
        response += f"   To: {dest_display}\n\n"

        def find_stops_near(lat: float, lon: float, radius: float) -> List[Dict]:
            cursor.execute("SELECT stop_id, stop_name, stop_lat, stop_lon FROM stops")
            all_stops = cursor.fetchall()

            nearby = []
            for stop in all_stops:
                if stop["stop_lat"] and stop["stop_lon"]:
                    dist = haversine_distance(
                        lat, lon, stop["stop_lat"], stop["stop_lon"]
                    )
                    if dist <= radius:
                        nearby.append(
                            {
                                "id": stop["stop_id"],
                                "name": stop["stop_name"],
                                "lat": stop["stop_lat"],
                                "lon": stop["stop_lon"],
                                "distance_km": dist,
                            }
                        )
            return sorted(nearby, key=lambda x: x["distance_km"])

        def filter_endpoint_stops(
            stops: List[Dict[str, Any]],
            own_lat: float,
            own_lon: float,
            other_lat: float,
            other_lon: float,
        ) -> List[Dict[str, Any]]:
            """Keep stops that are plausibly attached to their intended endpoint."""
            endpoint_stops: List[Dict[str, Any]] = []
            for stop in stops:
                own_distance = float(
                    stop.get("distance_km")
                    or haversine_distance(own_lat, own_lon, stop["lat"], stop["lon"])
                )
                other_distance = haversine_distance(other_lat, other_lon, stop["lat"], stop["lon"])
                if own_distance <= other_distance + 0.05:
                    endpoint_stops.append(stop)
            return endpoint_stops

        routes_found = []
        origin_stops = []
        dest_stops = []
        final_radius = search_radius_km

        for r in [search_radius_km, search_radius_km * 2, search_radius_km * 3]:
            final_radius = r
            raw_origin_stops = find_stops_near(origin_lat, origin_lon, r)
            raw_dest_stops = find_stops_near(dest_lat, dest_lon, r)

            if not raw_origin_stops or not raw_dest_stops:
                continue

            origin_stops = filter_endpoint_stops(
                raw_origin_stops,
                origin_lat,
                origin_lon,
                dest_lat,
                dest_lon,
            )
            dest_stops = filter_endpoint_stops(
                raw_dest_stops,
                dest_lat,
                dest_lon,
                origin_lat,
                origin_lon,
            )
            shared_stop_ids = {s["id"] for s in origin_stops} & {s["id"] for s in dest_stops}
            if shared_stop_ids:
                origin_stops = [s for s in origin_stops if s["id"] not in shared_stop_ids]
                dest_stops = [s for s in dest_stops if s["id"] not in shared_stop_ids]
            if not origin_stops or not dest_stops:
                continue

            origin_ids = [s["id"] for s in origin_stops]
            dest_ids = [s["id"] for s in dest_stops]

            ph_o = ",".join(["?" for _ in origin_ids])
            ph_d = ",".join(["?" for _ in dest_ids])

            sql = f"""
                SELECT DISTINCT r.route_id, r.route_short_name, r.route_long_name
                FROM routes r
                WHERE r.route_id IN (
                    SELECT t.route_id
                    FROM stop_times st_o
                    JOIN stop_times st_d ON st_o.trip_id = st_d.trip_id
                        AND st_o.stop_sequence < st_d.stop_sequence
                    JOIN trips t ON st_o.trip_id = t.trip_id
                    WHERE st_o.stop_id IN ({ph_o})
                      AND st_d.stop_id IN ({ph_d})
                )
            """
            cursor.execute(sql, origin_ids + dest_ids)
            routes_found = cursor.fetchall()

            if routes_found:
                break

        if not routes_found:
            conn.close()
            response += f"No direct Carris route found (radius: {final_radius}km).\n"
            response += "   Suggestions:\n"
            response += "   - Use Metro (faster for long distances)\n"
            response += (
                "   - Check connecting hubs like Entrecampos, Rossio or Cais do Sodré\n"
            )
            return response

        unique_routes: Dict[str, Dict[str, Any]] = {}
        for route in routes_found:
            unique_routes.setdefault(route["route_short_name"], dict(route))

        response += f"✅ **Direct routes found:** {len(unique_routes)}\n\n"

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        active_services = _get_active_services(conn, now)
        freshness_note = _build_gtfs_rt_freshness_note()
        if freshness_note:
            response += freshness_note + "\n\n"

        trams = [r for r in unique_routes.values() if r["route_short_name"].endswith("E")]
        buses = [r for r in unique_routes.values() if not r["route_short_name"].endswith("E")]
        origin_stop_by_id = {str(s["id"]): s for s in origin_stops}
        dest_stop_by_id = {str(s["id"]): s for s in dest_stops}

        def get_route_stop_hint(route_short_name: str) -> Dict[str, str]:
            """Return the matched origin/destination stops for a direct route."""
            if not origin_stops or not dest_stops:
                return {}

            origin_ids_hint = [s["id"] for s in origin_stops]
            dest_ids_hint = [s["id"] for s in dest_stops]
            ph_origin = ",".join(["?" for _ in origin_ids_hint])
            ph_dest = ",".join(["?" for _ in dest_ids_hint])
            cursor.execute(
                f"""
                SELECT
                    st_o.stop_id AS origin_stop_id,
                    st_d.stop_id AS destination_stop_id,
                    t.trip_headsign AS headsign,
                    so.stop_name AS origin_stop_name,
                    sd.stop_name AS destination_stop_name
                FROM stop_times st_o
                JOIN stop_times st_d ON st_o.trip_id = st_d.trip_id
                    AND st_o.stop_sequence < st_d.stop_sequence
                JOIN trips t ON st_o.trip_id = t.trip_id
                JOIN routes r ON t.route_id = r.route_id
                JOIN stops so ON st_o.stop_id = so.stop_id
                JOIN stops sd ON st_d.stop_id = sd.stop_id
                WHERE r.route_short_name = ?
                  AND st_o.stop_id IN ({ph_origin})
                  AND st_d.stop_id IN ({ph_dest})
                ORDER BY st_o.stop_sequence, st_d.stop_sequence
                LIMIT 80
                """,
                [route_short_name, *origin_ids_hint, *dest_ids_hint],
            )
            rows = cursor.fetchall()
            if not rows:
                return {}

            def stop_pair_score(row: sqlite3.Row) -> float:
                origin_stop = origin_stop_by_id.get(str(row["origin_stop_id"]), {})
                dest_stop = dest_stop_by_id.get(str(row["destination_stop_id"]), {})
                return float(origin_stop.get("distance_km", 99.0)) + float(
                    dest_stop.get("distance_km", 99.0)
                )

            return dict(min(rows, key=stop_pair_score))

        def format_final_walk_line(stop_hint: Dict[str, str]) -> str:
            """Return a final walking estimate from the alighting stop to destination."""
            hinted_dest = dest_stop_by_id.get(str(stop_hint.get("destination_stop_id")))
            if not hinted_dest:
                return ""
            distance_km = haversine_distance(
                dest_lat,
                dest_lon,
                hinted_dest["lat"],
                hinted_dest["lon"],
            )
            if distance_km > 1.2:
                return ""
            walking_minutes = max(1, round(distance_km * 12))
            return f"     Caminhada final: ~{walking_minutes} min até {dest_display}.\n"

        def format_initial_walk_line(stop_hint: Dict[str, str]) -> str:
            """Return an initial walking estimate from origin to boarding stop."""
            hinted_origin = origin_stop_by_id.get(str(stop_hint.get("origin_stop_id")))
            if not hinted_origin:
                return ""
            distance_km = haversine_distance(
                origin_lat,
                origin_lon,
                hinted_origin["lat"],
                hinted_origin["lon"],
            )
            if distance_km <= 0.12 or distance_km > 1.6:
                return ""
            walking_minutes = max(1, round(distance_km * 12))
            origin_stop_name = stop_hint.get("origin_stop_name") or hinted_origin.get("name") or "paragem"
            return f"     Caminhada inicial: ~{walking_minutes} min até {origin_stop_name}.\n"

        def format_route_line(r: Dict[str, Any]) -> str:
            """Format a single route entry with direction-safe departures and RT hints."""
            stop_hint = get_route_stop_hint(r["route_short_name"])
            route_origin_stops = origin_stops
            route_dest_stops = dest_stops
            if stop_hint:
                hinted_origin = origin_stop_by_id.get(str(stop_hint.get("origin_stop_id")))
                hinted_dest = dest_stop_by_id.get(str(stop_hint.get("destination_stop_id")))
                if hinted_origin and hinted_dest:
                    route_origin_stops = [hinted_origin]
                    route_dest_stops = [hinted_dest]

            departures = _get_directional_departures_for_route(
                conn=conn,
                origin_stops=route_origin_stops,
                dest_stops=route_dest_stops,
                route_short_name=r["route_short_name"],
                active_services=active_services,
                current_time=current_time,
                limit=4,
            )

            if not departures:
                headsign = stop_hint.get("headsign")
                route_summary = f"para {headsign}" if headsign else r["route_long_name"]
                line = f"   {r['route_short_name']}: {route_summary}\n"
                origin_stop_name = stop_hint.get("origin_stop_name")
                dest_stop_name = stop_hint.get("destination_stop_name")
                if origin_stop_name and dest_stop_name:
                    line += format_initial_walk_line(stop_hint)
                    line += f"     Stops: board at {origin_stop_name}; leave at stop {dest_stop_name}.\n"
                    line += format_final_walk_line(stop_hint)
                line += "     ℹ️ No upcoming departures were confirmed today at the matched origin stop.\n\n"
                return line

            first_dep = departures[0]
            line = f"   {r['route_short_name']}: para {first_dep['headsign']}\n"
            origin_stop_name = stop_hint.get("origin_stop_name")
            dest_stop_name = stop_hint.get("destination_stop_name")
            if origin_stop_name and dest_stop_name:
                line += format_initial_walk_line(stop_hint)
                line += f"     Stops: board at {origin_stop_name}; leave at stop {dest_stop_name}.\n"
                line += format_final_walk_line(stop_hint)
            shown_times = []
            for dep in departures[:3]:
                time_text = dep["estimated_departure"] if dep["is_realtime"] else dep["scheduled_departure"]
                if dep["is_realtime"]:
                    time_text = f"{time_text} {_format_delay_label(dep['delay_mins'])}"
                shown_times.append(time_text)

            line += f"     Next: {', '.join(shown_times)} (stop {first_dep['origin_stop_name']})\n"
            if first_dep.get("travel_mins"):
                line += f"     ~{first_dep['travel_mins']}min travel\n"
            line += "\n"
            return line

        if trams:
            response += "TRAMS\n" + "-" * 40 + "\n"
            for r in trams:
                response += format_route_line(r)

        if buses:
            response += "BUSES\n" + "-" * 40 + "\n"
            for r in buses[:5]:
                response += format_route_line(r)
            if len(buses) > 5:
                response += f"   ... and {len(buses) - 5} more routes\n\n"

        conn.close()
        return response

    except Exception as e:
        logger.error(f"Error finding routes: {e}")
        return f"Erro ao encontrar rotas: {e}"


@tool
def carris_get_realtime_vehicles(
    route_id: str = "",
    route_short_name: str = "",
    vehicle_type: str = "",
) -> str:
    """
    Gets real-time Carris vehicle positions from official GTFS-RT feed.

    Args:
        route_id: Filter by route short name (e.g., "28E", "15E", "732") - optional
        route_short_name: Alias for route_id, kept for compatibility with existing callers/tests.
        vehicle_type: Filter by "tram" or "bus" - optional

    Returns:
        Real-time vehicle positions with route and location info.
    """
    vehicles = fetch_gtfs_rt_vehicles()

    if not vehicles:
        if not GTFS_RT_AVAILABLE:
            return (
                "GTFS-RT library not installed. "
                "Run: pip install gtfs-realtime-bindings"
            )
        return (
            "Real-time vehicle data is currently unavailable. "
            f"The system attempted to fetch data from the Carris GTFS-RT feed "
            f"({REALTIME_MAX_RETRIES} attempts, {REALTIME_TIMEOUT}s timeout each) "
            f"but the endpoint did not respond in time. "
            f"This can happen during periods of high demand or maintenance. "
            f"Please try again in a few minutes. "
            f"Scheduled timetable information remains available via other tools."
        )

    conn = _get_db_connection()
    if not conn:
        return "Base de dados indisponível para enriquecer dados."

    try:
        enriched = [enrich_vehicle_with_static_data(v, conn) for v in vehicles]
        conn.close()

        filtered = enriched

        selected_route = route_short_name or route_id
        if selected_route:
            route_upper = selected_route.upper()
            filtered = [
                v
                for v in filtered
                if v.get("route_short_name", "").upper() == route_upper
            ]

        if vehicle_type:
            type_lower = vehicle_type.lower()
            if type_lower in ["tram", "elétrico", "eletrico"]:
                filtered = [v for v in filtered if v.get("is_tram", False)]
            elif type_lower in ["bus", "autocarro"]:
                filtered = [v for v in filtered if not v.get("is_tram", False)]

        if not filtered:
            filter_msg = ""
            if selected_route:
                filter_msg += f" rota={selected_route}"
            if vehicle_type:
                filter_msg += f" tipo={vehicle_type}"
            return f"Nenhum veículo encontrado com filtros:{filter_msg}"

        response = "Veículos Carris em Tempo Real\n"
        response += "=" * 55 + "\n"

        feed_time = filtered[0].get("feed_timestamp", 0)
        if feed_time:
            response += f"Dados de: {datetime.fromtimestamp(feed_time).strftime('%H:%M:%S')}\n\n"
        freshness_note = _build_gtfs_rt_freshness_note()
        if freshness_note:
            response += freshness_note + "\n\n"

        trams = [v for v in filtered if v.get("is_tram", False)]
        buses = [v for v in filtered if not v.get("is_tram", False)]

        def format_vehicle(v: Dict) -> str:
            route = v.get("route_short_name") or "Unknown line"
            headsign = _clean_carris_headsign(v.get("trip_headsign") or v.get("route_long_name") or "direction unavailable")
            lat = v.get("latitude", 0)
            lon = v.get("longitude", 0)
            stop = v.get("stop_name", "Em trânsito")
            plate = v.get("license_plate", "")
            status = v.get("current_status", "UNKNOWN")

            status_text = (
                "Em trânsito"
                if status == "IN_TRANSIT_TO"
                else "Parado"
                if status == "STOPPED_AT"
                else "A chegar"
            )

            line = f"{route} -> {_truncate_display_text(headsign, 34)} [{status_text}]\n"
            line += f"   GPS: {lat:.5f}, {lon:.5f}"
            if stop and stop != "Em trânsito":
                line += f" | Próxima paragem: {_truncate_display_text(stop, 28)}"
            if plate:
                line += f"\n   Matrícula: {plate}"
            return line + "\n"

        if trams:
            response += "ELÉTRICOS\n" + "-" * 40 + "\n"
            for v in trams[:15]:
                response += format_vehicle(v)
            if len(trams) > 15:
                response += f"   ... e mais {len(trams) - 15} elétricos\n"
            response += "\n"

        if buses:
            response += "AUTOCARROS\n" + "-" * 40 + "\n"
            for v in buses[:20]:
                response += format_vehicle(v)
            if len(buses) > 20:
                response += f"   ... e mais {len(buses) - 20} autocarros\n"
            response += "\n"

        response += f"Total: {len(trams)} elétricos + {len(buses)} autocarros = {len(filtered)} veículos\n"
        return response

    except Exception as e:
        logger.error(f"Error getting realtime vehicles: {e}")
        return f"Erro ao obter veículos: {e}"


@tool
def carris_vehicle_eta(route_short_name: str, stop_name: str) -> str:
    """
    Calculates estimated arrival time for vehicles of a specific route at a stop.

    This is the BEST tool for "when will bus/tram X arrive at stop Y".
    Combines real-time vehicle positions with static schedule for accurate ETAs.

    Args:
        route_short_name: Route number (e.g., "28E", "15E", "732")
        stop_name: Stop name or partial name

    Returns:
        Detailed ETA information with vehicle positions and delays.
    """
    conn = _get_db_connection()
    if not conn:
        return "Base de dados indisponível."

    try:
        cursor = conn.cursor()

        stops = _search_stop_rows(conn, stop_name, limit=5)
        if not stops:
            conn.close()
            return f"Paragem '{stop_name}' não encontrada."

        target_stop = stops[0]
        target_stop_id = target_stop["stop_id"]
        target_stop_name = target_stop["stop_name"]

        vehicles = get_vehicles_for_route(route_short_name)

        if not vehicles:
            # No real-time data: check if the route even exists before giving up
            cursor.execute(
                "SELECT route_short_name FROM routes WHERE route_short_name = ?",
                (route_short_name,),
            )
            route_exists = cursor.fetchone()

            if not route_exists:
                conn.close()
                return (
                    f"Route '{route_short_name}' not found in Carris network. "
                    f"Use carris_get_routes to see available routes."
                )

            # Route exists but no vehicles: fall back to scheduled data
            vehicle_icon = "🚋" if route_short_name.upper().endswith("E") else "🚌"
            response = f"### {vehicle_icon} **{route_short_name} at {target_stop_name}**\n\n"
            response += "- ℹ️ **Live Arrival Estimate:** no active real-time vehicle was detected for this line right now.\n"
            response += "- 🕒 **Scheduled fallback:**\n"

            now_dt = datetime.now()
            active_services = _get_active_services(conn, now_dt)
            if active_services:
                ph = ",".join(["?" for _ in active_services])
                cursor.execute(
                    f"""
                    SELECT st.departure_time, t.trip_headsign
                    FROM stop_times st
                    JOIN trips t ON st.trip_id = t.trip_id
                    JOIN routes r ON t.route_id = r.route_id
                    WHERE st.stop_id = ? AND r.route_short_name = ?
                    AND t.service_id IN ({ph})
                    AND st.departure_time >= ?
                    ORDER BY st.departure_time LIMIT 5
                """,
                    (
                        target_stop_id,
                        route_short_name,
                        *active_services,
                        now_dt.strftime("%H:%M:%S"),
                    ),
                )
                deps = cursor.fetchall()
                for departure in deps:
                    departure_clock = _format_service_clock(
                        time_str_to_minutes(departure["departure_time"])
                    )
                    headsign = _clean_carris_headsign(
                        departure["trip_headsign"] or ""
                    )
                    response += f"    - {departure_clock}"
                    if headsign:
                        response += f" → {headsign}"
                    response += "\n"
                if not deps:
                    response += "    - No more departures scheduled for today.\n"
            else:
                response += "   No active services found for today.\n"

            conn.close()
            return response

        vehicle_icon = "🚋" if route_short_name.upper().endswith("E") else "🚌"
        response = f"### {vehicle_icon} **{route_short_name} at {target_stop_name}**\n\n"
        response += f"- 🕒 **Updated:** {datetime.now().strftime('%H:%M')}\n"
        freshness_note = _build_gtfs_rt_freshness_note()
        if freshness_note:
            response += f"- {freshness_note}\n"

        etas = []
        for v in vehicles:
            eta_info = get_vehicle_eta_at_stop(v, target_stop_id, conn)
            if eta_info:
                etas.append(
                    {
                        **eta_info,
                        "vehicle_id": v.get("vehicle_id"),
                        "license_plate": v.get("license_plate"),
                        "current_stop": v.get("stop_name", "Em trânsito"),
                    }
                )

        if not etas:
            response += "- ℹ️ **Live Arrival Estimate:** no active vehicle is currently matched to this stop.\n"
            response += "- 🕒 **Scheduled fallback:**\n"

            now = datetime.now()
            active_services = _get_active_services(conn, now)

            if active_services:
                ph = ",".join(["?" for _ in active_services])
                cursor.execute(
                    f"""
                    SELECT st.departure_time, t.trip_headsign
                    FROM stop_times st
                    JOIN trips t ON st.trip_id = t.trip_id
                    JOIN routes r ON t.route_id = r.route_id
                    WHERE st.stop_id = ? AND r.route_short_name = ? AND t.service_id IN ({ph})
                    AND st.departure_time >= ?
                    ORDER BY st.departure_time LIMIT 5
                """,
                    (
                        target_stop_id,
                        route_short_name,
                        *active_services,
                        now.strftime("%H:%M:%S"),
                    ),
                )

                deps = cursor.fetchall()
                for departure in deps:
                    departure_clock = _format_service_clock(
                        time_str_to_minutes(departure["departure_time"])
                    )
                    headsign = _clean_carris_headsign(
                        departure["trip_headsign"] or ""
                    )
                    response += f"    - {departure_clock}"
                    if headsign:
                        response += f" → {headsign}"
                    response += "\n"

            conn.close()
            return response

        etas.sort(key=lambda x: time_str_to_minutes(x["estimated_arrival"]))

        response += f"- ✅ **Live vehicles approaching:** {len(etas)}\n"

        for eta in etas[:5]:
            delay_str = ""
            if eta["current_delay_mins"] > 0:
                delay_str = f" (atrasado +{eta['current_delay_mins']} min)"
            elif eta["current_delay_mins"] < 0:
                delay_str = f" (adiantado {abs(eta['current_delay_mins'])} min)"

            minutes_until = max(_minutes_until_clock_time(eta["estimated_arrival"]) or 0, 0)
            response += f"**{vehicle_icon} {route_short_name}**\n"
            response += f"    - ⏱️ **Estimated Arrival:** {eta['estimated_arrival']} ({minutes_until} min){delay_str}\n"
            response += f"    - 📍 **Current position:** {eta['current_stop']}\n"
            response += f"    - 🚏 **Stops remaining:** {eta['stops_remaining']}\n"

            if eta.get("license_plate"):
                response += f"    - 🚗 **Vehicle:** {eta['license_plate']}\n"

            response += f"    - 🧭 **Scheduled travel from current position:** ~{eta['scheduled_travel_mins']} min\n\n"

        if len(etas) > 5:
            response += f"   ... e mais {len(etas) - 5} veículos\n"

        conn.close()
        return response

    except Exception as e:
        logger.error(f"Error calculating ETA: {e}")
        return f"Erro ao calcular tempo estimado de chegada: {e}"


@tool
def carris_get_service_frequency(
    route_short_name: str,
    stop_name: Optional[str] = None,
) -> str:
    """
    Estimates bus/tram service frequency (headway) for a Carris route.
    Calculates average time between departures by time window (morning, midday, afternoon, evening).
    Uses GTFS stop_times data since frequencies.txt is not available.

    Args:
        route_short_name: Route number (e.g., '28E', '15E', '738', '714').
        stop_name: Optional stop name to check frequency at a specific stop.
                   If not provided, uses the first stop of the route.

    Returns:
        str: Formatted frequency information by time window.

    Examples:
        >>> carris_get_service_frequency("28E")
        >>> carris_get_service_frequency("15E", stop_name="Praça da Figueira")
    """
    conn = _get_db_connection()
    if not conn:
        return "Carris database unavailable."

    try:
        cursor = conn.cursor()

        # Find route
        cursor.execute(
            "SELECT route_id, route_short_name, route_long_name FROM routes WHERE route_short_name = ?",
            (route_short_name,),
        )
        route = cursor.fetchone()
        if not route:
            conn.close()
            return f"Route '{route_short_name}' not found in Carris data."

        route_id = route["route_id"]

        # Get active services for today
        active_services = _get_active_services(conn)
        if not active_services:
            conn.close()
            return "No active services found for today."

        ph_s = ",".join(["?" for _ in active_services])

        # Determine which stop to analyze
        if stop_name:
            cursor.execute(
                "SELECT stop_id, stop_name FROM stops WHERE stop_name LIKE ? LIMIT 1",
                (f"%{stop_name}%",),
            )
            stop_row = cursor.fetchone()
            if not stop_row:
                conn.close()
                return f"Stop '{stop_name}' not found."
            stop_id = stop_row["stop_id"]
            stop_display = stop_row["stop_name"]
        else:
            # Use first stop on the route (stop_sequence = 1 or min)
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
                return f"No scheduled trips found for route '{route_short_name}' today."
            stop_id = stop_row["stop_id"]
            stop_display = stop_row["stop_name"]

        # Get all departure times at this stop for this route today
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
            return f"No departures found for route '{route_short_name}' at '{stop_display}' today."

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
            ("🌃 Night (23:00-05:59)", 1380, 1800),
        ]

        icon = "🚋" if route_short_name.upper().endswith("E") else "🚌"
        response = f"### {icon} **Route {route_short_name} service frequency**\n\n"
        response += f"- 📍 **Stop:** {stop_display}\n"
        response += f"- 📊 **Total departures today (Paragens):** {len(departures)}\n\n"

        for window_name, start_min, end_min in windows:
            window_deps = [d for d in departures if start_min <= d < end_min]

            if len(window_deps) < 2:
                if len(window_deps) == 1:
                    t = window_deps[0]
                    response += f"**{window_name}**\n"
                    response += f"    - 🕒 One departure: {_format_service_clock(t)}\n"
                else:
                    response += f"**{window_name}**\n"
                    response += "    - No service\n"
                continue

            # Calculate headways between consecutive departures
            headways = [window_deps[i + 1] - window_deps[i] for i in range(len(window_deps) - 1)]
            avg_headway = sum(headways) / len(headways)
            min_headway = min(headways)
            max_headway = max(headways)

            first_dep = window_deps[0]
            last_dep = window_deps[-1]

            response += f"**{window_name}**\n"
            response += f"    - ⏱️ **Avg frequency:** {avg_headway:.0f} min · **Min/Max:** {min_headway}-{max_headway} min\n"
            response += f"    - 🕒 **First:** {_format_service_clock(first_dep)} · **Last:** {_format_service_clock(last_dep)} · {icon} **Departures:** {len(window_deps)}\n\n"

        response += "⚠️ Frequencies can vary with traffic and service changes.\n"

        return response

    except Exception as e:
        logger.error(f"Error calculating frequency: {e}")
        return f"Error calculating frequency: {e}"


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    import sys
    import time as time_module
    with contextlib.suppress(AttributeError):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    # Argument Parsing
    parser = argparse.ArgumentParser(description="Carris API Tools Test Suite")
    parser.add_argument(
        "--rebuild-db", action="store_true", help="Force database rebuild"
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("\033[1m🧪 COMPREHENSIVE TEST: Carris Urban API Tools\033[0m")
    print("=" * 70)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🗄️ Database: {CARRIS_DB_PATH}")
    print(f"📡 GTFS-RT Available: {GTFS_RT_AVAILABLE}")
    print(f"🔄 Rebuild DB: {args.rebuild_db}")
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
            # Truncate long outputs for readability
            # if isinstance(result, str) and len(result) > 800:
            #     print(result[:800] + "\n\n... (truncated for readability)")
            # elif result is not None:
            print(result)
            test_results["passed"] += 1
            print("\n\033[1;32m✅ PASSED\033[0m")
            return result
        except Exception as e:
            print(f"\n\033[1;31m❌ FAILED: {str(e)}\033[0m")
            test_results["failed"] += 1
            return None

    # =========================================================================
    # DATABASE SETUP TESTS
    # =========================================================================
    # TEST 1: Database Setup (Conditional Rebuild)
    def _test_database_setup():
        print(f"Ensuring database (force_update={args.rebuild_db})...")
        manager = CarrisGTFSManager()
        start = time_module.time()
        success = manager.ensure_database(force_update=args.rebuild_db)
        elapsed = time_module.time() - start

        if not success:
            raise AssertionError("Database setup failed")

        db_size = os.path.getsize(CARRIS_DB_PATH) / (1024 * 1024)
        return f"Database ready in {elapsed:.1f}s ({db_size:.1f} MB)"

    run_test("Database Setup", _test_database_setup)

    # =========================================================================
    # SCHEMA VERIFICATION TESTS
    # =========================================================================
    # TEST 2: Primary Keys Verification
    def _test_schema_pks():
        conn = sqlite3.connect(CARRIS_DB_PATH)
        cur = conn.cursor()

        expected_pks = {
            "agency": ["agency_id"],
            "routes": ["route_id"],
            "stops": ["stop_id"],
            "trips": ["trip_id"],
            "stop_times": ["trip_id", "stop_sequence"],
            "calendar": ["service_id"],
            "calendar_dates": ["service_id", "date"],
            "shapes": ["shape_id", "shape_pt_sequence"],
        }

        results = []
        all_ok = True

        for table, expected in expected_pks.items():
            cur.execute(f"PRAGMA table_info({table})")
            pk_cols = [r[1] for r in cur.fetchall() if r[5] > 0]

            if pk_cols == expected:
                results.append(f"   ✅ {table}: PK = {pk_cols}")
            else:
                results.append(f"   ❌ {table}: Expected {expected}, got {pk_cols}")
                all_ok = False

        conn.close()

        if not all_ok:
            raise AssertionError("Primary key verification failed")

        return "\n".join(results)

    run_test("Schema Verification (PRIMARY KEYs)", _test_schema_pks)

    # TEST 3: Index Verification
    def _test_schema_indexes():
        conn = sqlite3.connect(CARRIS_DB_PATH)
        cur = conn.cursor()

        expected_indexes = [
            "idx_routes_short_name",
            "idx_routes_type",
            "idx_stops_name",
            "idx_stops_coords",
            "idx_trips_route",
            "idx_trips_service",
            "idx_trips_route_service",
            "idx_stop_times_stop",
            "idx_stop_times_stop_time",
            "idx_stop_times_trip_seq",
            "idx_calendar_dates",
            "idx_calendar_dates_date",
            "idx_shapes_id",
        ]

        cur.execute(
            'SELECT name FROM sqlite_master WHERE type="index" AND name NOT LIKE "sqlite_%"'
        )
        actual_indexes = [r[0] for r in cur.fetchall()]
        conn.close()

        missing = set(expected_indexes) - set(actual_indexes)

        if missing:
            raise AssertionError(f"Missing indexes: {missing}")

        return f"   ✅ All {len(expected_indexes)} expected indexes present"

    run_test("Index Verification", _test_schema_indexes)

    # =========================================================================
    # TOOL TESTS
    # =========================================================================

    def _resolve_stop_id_from_db(query: str) -> Optional[str]:
        """Resolve a stop_id directly from the Carris DB for deterministic test setup."""
        conn = _get_db_connection()
        if not conn:
            return None
        try:
            rows = _search_stop_rows(conn, query, limit=1)
            return str(rows[0]["stop_id"]) if rows else None
        finally:
            conn.close()

    # TEST 4: Get Stops (Prerequisite for other tests)
    def _test_get_stops_rossio():
        """Retrieve Rossio stops to use IDs in next tests."""
        print("Finding all Rossio stops (unlimited)...")
        result = carris_get_stops.invoke(
            {"query": "Rossio"}
        )  # No limit implicit > all results
        if "não encontrada" in result:
            raise AssertionError("Could not find Rossio stops")

        # Resolve stop IDs from DB
        stop_id = _resolve_stop_id_from_db("Rossio")
        if stop_id:
            print(result)
            return stop_id
        raise AssertionError("Could not extract stop ID from result")

    stop_id_rossio = None
    stops_result = run_test("Tool: carris_get_stops('Rossio')", _test_get_stops_rossio)
    if stops_result:
        stop_id_rossio = stops_result

    # TEST 5: carris_get_next_departures (Static Schedule)
    if stop_id_rossio:
        print("\nNote: carris_get_next_departures uses the STATIC GTFS schedule.")
        print(
            "      It answers 'What is the planned time?' (useful for future planning)."
        )
        run_test(
            f"Tool: carris_get_next_departures (Stop {stop_id_rossio})",
            carris_get_next_departures.invoke,
            {"stop_id": stop_id_rossio, "limit": 3},
        )

        # TEST 6: carris_get_next_departures with START TIME & FILTER
        future_time = (datetime.now() + timedelta(hours=2)).strftime("%H:%M")
        run_test(
            f"Tool: carris_get_next_departures (Stop {stop_id_rossio}, Time {future_time}, Line Filter)",
            carris_get_next_departures.invoke,
            {
                "stop_id": stop_id_rossio,
                "start_time": future_time,
                "route_short_name": "759",
            },
        )
    else:
        print("⚠️ Skipping next_departures tests due to missing stop ID")

    # TEST 7: carris_find_routes_between (Complex Routing)
    run_test(
        "Tool: carris_find_routes_between ('Cais Sodré' -> 'Belém')",
        carris_find_routes_between.invoke,
        {"origin": "Cais Sodré", "destination": "Belém"},
    )

    # TEST 8: carris_get_realtime_vehicles
    run_test(
        "Tool: carris_get_realtime_vehicles (Tram 28E)",
        carris_get_realtime_vehicles.invoke,
        {"route_id": "28E"},
    )

    # TEST 9: carris_vehicle_eta (Edge Case: Invalid Line)
    run_test(
        "Edge Case: carris_vehicle_eta (Invalid Line)",
        carris_vehicle_eta.invoke,
        {"route_short_name": "999X", "stop_name": "Rossio"},
    )

    # TEST 10: "A que horas passa o 758 no Rato?"
    # carris_get_next_departures requires stop_id (not stop_name)
    # First resolve stop name to ID, then query
    def _test_next_departures_by_name():
        stops_result = carris_get_stops.invoke({"query": "Rato"})
        print(stops_result)

        # Resolve stop IDs from DB to avoid brittle markdown parsing.
        stop_id = _resolve_stop_id_from_db("Rato")
        if stop_id:
            return carris_get_next_departures.invoke(
                {"stop_id": stop_id, "route_short_name": "758"}
            )
        return "Could not resolve stop name 'Rato' to stop_id"

    run_test(
        "Tool: carris_get_next_departures (Stop Rato, Line 758)",
        _test_next_departures_by_name,
    )

    # TEST 11: "Onde está o 28E agora?"
    run_test(
        "Tool: carris_get_realtime_vehicles (Tram 28E)",
        carris_get_realtime_vehicles.invoke,
        {"route_id": "28E"},
    )

    # TEST 12: "Como chego da Estação do Oriente ao Castelo?"
    run_test(
        "Tool: carris_find_routes_between (Estação do Oriente -> Castelo)",
        carris_find_routes_between.invoke,
        {"origin": "Estação do Oriente", "destination": "Castelo"},
    )

    # TEST 13: "Quantos minutos faltam para o próximo 714 na Praça da Figueira?"
    run_test(
        "Tool: carris_vehicle_eta (Praça da Figueira, Line 714)",
        carris_vehicle_eta.invoke,
        {"stop_name": "Praça da Figueira", "route_short_name": "714"},
    )

    # =========================================================================
    # DIRECTION-AWARE ROUTING TESTS
    # =========================================================================

    # TEST 14: Direction-aware routing (origin BEFORE destination in trip)
    def _test_direction_fix():
        result = carris_find_routes_between.invoke(
            {"origin": "Cais do Sodré", "destination": "Belém"}
        )
        if "não encontrada" in result.lower() and "erro" in result.lower():
            raise AssertionError("Route not found - direction fix may have broken routing")
        if "Linha" in result or "Route" in result or "rota" in result.lower():
            # return f"✅ Direction-aware routing working\n{result[:500]}"
            return f"✅ Direction-aware routing working\n{result}"
        # return result[:500]
        return result

    run_test("Direction Routing: Cais do Sodré → Belém (forward)", _test_direction_fix)

    # TEST 15: Reverse direction test
    def _test_reverse_direction():
        result = carris_find_routes_between.invoke(
            {"origin": "Belém", "destination": "Cais do Sodré"}
        )
        if "não encontrada" in result.lower() and "erro" in result.lower():
            raise AssertionError("Reverse route not found")
        # return f"✅ Reverse routing working\n{result[:500]}"
        return f"✅ Reverse routing working\n{result}"

    run_test("Direction Routing: Belém → Cais do Sodré (reverse)", _test_reverse_direction)

    # TEST 16: SQL direction constraint verification
    def _test_sql_direction_constraint():
        conn = sqlite3.connect(CARRIS_DB_PATH)
        cur = conn.cursor()
        # Verify the query uses proper direction with st_o.stop_sequence < st_d.stop_sequence
        # by checking a known route (15E goes Praça da Figueira → Belém)
        cur.execute("""
            SELECT DISTINCT r.route_short_name, r.route_long_name
            FROM routes r
            JOIN trips t ON r.route_id = t.route_id
            JOIN stop_times st_o ON t.trip_id = st_o.trip_id
            JOIN stop_times st_d ON t.trip_id = st_d.trip_id
            JOIN stops s_o ON st_o.stop_id = s_o.stop_id
            JOIN stops s_d ON st_d.stop_id = s_d.stop_id
            WHERE s_o.stop_name LIKE '%Figueira%'
              AND s_d.stop_name LIKE '%Belém%'
              AND st_o.stop_sequence < st_d.stop_sequence
            LIMIT 5
        """)
        rows = cur.fetchall()
        conn.close()

        if not rows:
            raise AssertionError("No routes found with direction constraint (stop_sequence ordering)")

        return "Routes Figueira→Belém with direction constraint:\n" + \
               "\n".join(f"   {r[0]}: {r[1]}" for r in rows)

    run_test("SQL Direction Constraint Verification", _test_sql_direction_constraint)

    # =========================================================================
    # FREQUENCY TOOL TESTS
    # =========================================================================

    # TEST 17: carris_get_service_frequency - Tram 28E
    run_test(
        "carris_get_service_frequency('28E')",
        carris_get_service_frequency.invoke,
        {"route_short_name": "28E"},
    )

    # TEST 18: carris_get_service_frequency - Bus 758
    run_test(
        "carris_get_service_frequency('758')",
        carris_get_service_frequency.invoke,
        {"route_short_name": "758"},
    )

    # TEST 19: carris_get_service_frequency with specific stop
    run_test(
        "carris_get_service_frequency('15E', stop='Belém')",
        carris_get_service_frequency.invoke,
        {"route_short_name": "15E", "stop_name": "Belém"},
    )

    # TEST 20: Frequency output format validation
    def _test_frequency_format():
        result = carris_get_service_frequency.invoke({"route_short_name": "28E"})
        print(result)

        lower_result = result.lower()
        checks = {
            "has_title": "28E" in result,
            "has_morning": "Morning" in result or "morning" in result.lower(),
            "has_frequency": "min" in lower_result,
            "has_count": any(
                marker in lower_result
                for marker in ("passagens", "paragens", "departures", "total")
            ),
        }
        errors = [k for k, v in checks.items() if not v]
        if errors:
            raise AssertionError(f"Missing in frequency output: {errors}")
        return f"✅ Frequency output format valid (checks: {list(checks.keys())})"

    run_test("Frequency Output Format Validation", _test_frequency_format)

    # TEST 21: Frequency - invalid route (edge case)
    run_test(
        "carris_get_service_frequency('999Z') - Invalid Route",
        carris_get_service_frequency.invoke,
        {"route_short_name": "999Z"},
    )

    # TEST 22: Direction-safe Rossio departures for line 732
    if stop_id_rossio:
        run_test(
            "Regression: carris_get_next_departures (Rossio, line 732)",
            carris_get_next_departures.invoke,
            {"stop_id": stop_id_rossio, "route_short_name": "732", "limit": 8},
        )

        run_test(
            "Regression: carris_get_arrivals (Rossio)",
            carris_get_arrivals.invoke,
            {"stop_id": stop_id_rossio, "limit": 8},
        )

    # TEST 24: Accent-insensitive ETA lookup
    run_test(
        "Regression: carris_vehicle_eta (Praca da Figueira, line 714)",
        carris_vehicle_eta.invoke,
        {"stop_name": "Praca da Figueira", "route_short_name": "714"},
    )

    # TEST 25: route_short_name realtime vehicle lookup
    run_test(
        "Regression: carris_get_realtime_vehicles (route_short_name='15E')",
        carris_get_realtime_vehicles.invoke,
        {"route_short_name": "15E"},
    )

    # TEST 26: Rossio -> Belém urban route query
    run_test(
        "Regression: carris_find_routes_between (Rossio -> Belém)",
        carris_find_routes_between.invoke,
        {"origin": "Rossio", "destination": "Belém"},
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("\033[1m📊 TEST SUMMARY\033[0m")
    print("=" * 70)
    print(
        f"\033[1;32m✅ Passed: {test_results['passed']}/{test_results['total']}\033[0m"
    )
    if test_results["failed"] > 0:
        print(
            f"\033[1;31m❌ Failed: {test_results['failed']}/{test_results['total']}\033[0m"
        )
    else:
        print("\033[1;36m✨ All tests completed successfully.\033[0m")
    print("=" * 70 + "\n")
