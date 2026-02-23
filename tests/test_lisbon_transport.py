# ==========================================================================
# Master Thesis
#   - André Filipe Gomes Silvestre, 20240502
#
# Comprehensive Test Suite for Lisbon Transport APIs
# This script validates all transport data sources and tools for the
# Multi-Agent Tourist Itinerary System.
#
# APIs Tested:
#   - Metro de Lisboa (Official API with OAuth2)
#   - Carris (Urban Lisbon buses and trams via GTFS + GTFS-RT)
#   - Carris Metropolitana (Suburban buses)
#   - CP Comboios de Portugal (Trains via GTFS + Real-time)
#   - IPMA (Weather data)
#
# Run: python tests/test_lisbon_transport.py
# ==========================================================================

# Required libraries:
# pip install requests langchain-core

import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


# ==========================================================================
# Test Result Tracking
# ==========================================================================

class TestResults:
    """Tracks test results with pass/fail counts."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.start_time = time.time()
    
    def add_pass(self, test_name: str):
        self.passed += 1
        print(f"   \033[1;32m✅ PASS\033[0m: {test_name}")
    
    def add_fail(self, test_name: str, reason: str):
        self.failed += 1
        self.errors.append(f"{test_name}: {reason}")
        print(f"   \033[1;31m❌ FAIL\033[0m: {test_name}")
        print(f"      Reason: {reason}")
    
    def add_warning(self, test_name: str, message: str):
        self.warnings.append(f"{test_name}: {message}")
        print(f"   \033[1;33m⚠️ WARN\033[0m: {test_name}")
        print(f"      {message}")
    
    def summary(self) -> str:
        elapsed = time.time() - self.start_time
        total = self.passed + self.failed
        
        result = "\n" + "=" * 70 + "\n"
        result += "\033[1m📊 TEST SUMMARY\033[0m\n"
        result += "=" * 70 + "\n\n"
        
        if self.failed == 0:
            result += f"\033[1;32m✅ ALL TESTS PASSED: {self.passed}/{total}\033[0m\n"
        else:
            result += f"\033[1;31m❌ TESTS FAILED: {self.failed}/{total}\033[0m\n"
            result += f"\033[1;32m✅ Tests passed: {self.passed}\033[0m\n"
        
        if self.warnings:
            result += f"\033[1;33m⚠️ Warnings: {len(self.warnings)}\033[0m\n"
        
        result += f"\n⏱️ Total time: {elapsed:.2f}s\n"
        
        if self.errors:
            result += "\n\033[1;31mErrors:\033[0m\n"
            for error in self.errors:
                result += f"   • {error}\n"
        
        if self.warnings:
            result += "\n\033[1;33mWarnings:\033[0m\n"
            for warning in self.warnings:
                result += f"   • {warning}\n"
        
        return result


# ==========================================================================
# Test Functions
# ==========================================================================

def test_metro_api(results: TestResults) -> None:
    """Tests Metro de Lisboa API functionality."""
    print("\n" + "-" * 70)
    print("\033[1m🚇 TEST: Metro de Lisboa API\033[0m")
    print("-" * 70)
    
    try:
        from tools.metrolisboa_api import (
            get_metro_status,
            get_all_metro_stations,
            get_metro_wait_time,
            find_nearest_metro,
            METRO_LINES,
            METRO_STATIONS
        )
        
        # Test 1: Line definitions
        if len(METRO_LINES) == 4:
            results.add_pass("Metro has 4 lines defined (amarela, azul, verde, vermelha)")
        else:
            results.add_fail("Metro line count", f"Expected 4, got {len(METRO_LINES)}")
        
        # Test 2: Station count
        if len(METRO_STATIONS) >= 50:
            results.add_pass(f"Metro has {len(METRO_STATIONS)} stations defined")
        else:
            results.add_fail("Metro station count", f"Expected >= 50, got {len(METRO_STATIONS)}")
        
        # Test 3: Get metro status (API call)
        status = get_metro_status.invoke({})
        if status and "Metro de Lisboa" in status:
            results.add_pass("get_metro_status returns valid response")
        else:
            results.add_fail("get_metro_status", "Invalid response format")
        
        # Test 4: Get all stations (API call)
        stations = get_all_metro_stations.invoke({})
        if stations and "Yellow Line" in stations:
            results.add_pass("get_all_metro_stations returns valid response")
        else:
            results.add_fail("get_all_metro_stations", "Invalid response format")
        
        # Test 5: Get wait time for a station
        wait_time = get_metro_wait_time.invoke({"station": "Campo Grande"})
        if wait_time and ("Wait Time" in wait_time or "Direction" in wait_time):
            results.add_pass("get_metro_wait_time returns wait times")
        else:
            results.add_warning("get_metro_wait_time", "API may be temporarily unavailable")
        
        # Test 6: Find nearest metro by coordinates (Colombo area)
        nearest = find_nearest_metro.invoke({"latitude": 38.7548, "longitude": -9.1889, "limit": 3})
        if nearest and "Nearest Metro" in nearest:
            results.add_pass("find_nearest_metro returns nearby stations")
        else:
            results.add_warning("find_nearest_metro", "Check response format")
        
    except ImportError as e:
        results.add_fail("Metro API Import", str(e))
    except Exception as e:
        results.add_fail("Metro API", str(e))


def test_carris_api(results: TestResults) -> None:
    """Tests Carris Urban API functionality."""
    print("\n" + "-" * 70)
    print("\033[1m🚌 TEST: Carris Urban API (Lisbon buses/trams)\033[0m")
    print("-" * 70)
    
    try:
        from tools.carris_api import (
            carris_get_stops,
            carris_get_routes,
            carris_get_next_departures,
            carris_find_routes_between,
            carris_get_realtime_vehicles,
            CarrisGTFSManager
        )
        
        # Test 1: GTFS Manager exists
        manager = CarrisGTFSManager()
        import pathlib
        db_path = pathlib.Path(manager.db_path) if isinstance(manager.db_path, str) else manager.db_path
        if db_path.exists():
            results.add_pass("Carris GTFS database exists")
        else:
            results.add_warning("Carris GTFS", "Database not found, will be created on first use")
            # Skip remaining tests if no database
            return
        
        # Test 2: Search stops (with timeout handling)
        try:
            stops = carris_get_stops.invoke({"query": "Rossio", "limit": 5})
            if stops and "Rossio" in str(stops):
                results.add_pass("carris_get_stops finds Rossio stops")
            else:
                results.add_fail("carris_get_stops", "Could not find Rossio stops")
        except Exception as e:
            results.add_warning("carris_get_stops", f"API timeout or error: {str(e)[:50]}")
        
        # Test 3: Get routes (with timeout handling)
        try:
            routes = carris_get_routes.invoke({"route_type": "tram", "limit": 5})
            if routes and ("28E" in routes or "Elétrico" in routes or "ELÉTRICO" in routes or "ELÉCTRICO" in routes):
                results.add_pass("carris_get_routes returns tram routes")
            else:
                results.add_fail("carris_get_routes", "Invalid response format")
        except Exception as e:
            results.add_warning("carris_get_routes", f"API timeout or error: {str(e)[:50]}")
        
        # Test 4: Get next departures
        try:
            departures = carris_get_next_departures.invoke({"stop_id": "908", "limit": 5})
            if departures and "Departures" in departures:
                results.add_pass("carris_get_next_departures returns schedule")
            else:
                results.add_fail("carris_get_next_departures", "Invalid response format")
        except Exception as e:
            results.add_warning("carris_get_next_departures", f"API error: {str(e)[:50]}")
        
        # Test 5: Find routes between locations
        try:
            route = carris_find_routes_between.invoke({"origin": "Rossio", "destination": "Belém"})
            if route and ("Route" in route or "direct" in route.lower() or "bus" in route.lower()):
                results.add_pass("carris_find_routes_between returns route options")
            else:
                results.add_warning("carris_find_routes_between", "No direct routes found (may be expected)")
        except Exception as e:
            results.add_warning("carris_find_routes_between", f"API error: {str(e)[:50]}")
        
        # Test 6: Real-time vehicles
        try:
            vehicles = carris_get_realtime_vehicles.invoke({"route_short_name": "28E"})
            if vehicles and ("28E" in vehicles or "Elétrico" in vehicles or "ELÉTRICO" in vehicles):
                results.add_pass("carris_get_realtime_vehicles tracks tram 28E")
            else:
                results.add_warning("carris_get_realtime_vehicles", "No vehicles found (may be off-hours)")
        except Exception as e:
            results.add_warning("carris_get_realtime_vehicles", f"API error: {str(e)[:50]}")
        
    except ImportError as e:
        results.add_fail("Carris API Import", str(e))
    except Exception as e:
        results.add_fail("Carris API", str(e))


def test_carris_metropolitana_api(results: TestResults) -> None:
    """Tests Carris Metropolitana API functionality."""
    print("\n" + "-" * 70)
    print("\033[1m🚌 TEST: Carris Metropolitana API (Suburban buses)\033[0m")
    print("-" * 70)
    
    try:
        from tools.carrismetropolitana_api import (
            get_carris_metropolitana_alerts,
            get_carris_metropolitana_stop_info,
            search_carris_metropolitana_lines,
            find_bus_routes,
            get_bus_realtime_locations,
            geocode_location
        )
        
        # Test 1: Get alerts (API changed to return list)
        alerts = get_carris_metropolitana_alerts.invoke({})
        if alerts and ("alert" in alerts.lower() or "no active" in alerts.lower()):
            results.add_pass("get_carris_metropolitana_alerts returns valid response")
        else:
            results.add_fail("get_carris_metropolitana_alerts", "Invalid response format")
        
        # Test 2: Search lines
        lines = search_carris_metropolitana_lines.invoke({"query": "Sintra"})
        if lines and "Sintra" in lines:
            results.add_pass("search_carris_metropolitana_lines finds Sintra lines")
        else:
            results.add_fail("search_carris_metropolitana_lines", "Could not find Sintra lines")
        
        # Test 3: Geocode location
        result = geocode_location("Colombo")
        if result and isinstance(result, dict) and "lat" in result and "lon" in result:
            lat, lon = result["lat"], result["lon"]
            if lat > 38 and lon < -9:
                results.add_pass(f"geocode_location returns valid coords for Colombo ({lat:.4f}, {lon:.4f})")
            else:
                results.add_warning("geocode_location", f"Coords out of range: ({lat}, {lon})")
        else:
            results.add_warning("geocode_location", f"Unexpected result: {result}")
        
        # Test 4: Find bus routes
        routes = find_bus_routes.invoke({"origin": "Sintra", "destination": "Lisboa"})
        if routes and ("route" in routes.lower() or "bus" in routes.lower()):
            results.add_pass("find_bus_routes returns route information")
        else:
            results.add_warning("find_bus_routes", "No routes found (may need refinement)")
        
    except ImportError as e:
        results.add_fail("Carris Metropolitana Import", str(e))
    except Exception as e:
        results.add_fail("Carris Metropolitana API", str(e))


def test_cp_api(results: TestResults) -> None:
    """Tests CP Trains API functionality."""
    print("\n" + "-" * 70)
    print("\033[1m🚆 TEST: CP Comboios de Portugal API\033[0m")
    print("-" * 70)
    
    try:
        from tools.cp_api import (
            get_train_status,
            search_cp_stations,
            get_train_schedule,
            get_cp_routes,
            initialize_cp_gtfs,
            CP_LINES,
            CP_KEY_STATIONS,
            CPGTFSManager
        )
        
        # Test 1: Line definitions
        if len(CP_LINES) >= 5:
            results.add_pass(f"CP has {len(CP_LINES)} lines defined (cascais, sintra, azambuja, norte, fertagus, sado)")
        else:
            results.add_fail("CP line count", f"Expected >= 5, got {len(CP_LINES)}")
        
        # Test 2: Key stations
        if len(CP_KEY_STATIONS) >= 10:
            results.add_pass(f"CP has {len(CP_KEY_STATIONS)} key stations defined")
        else:
            results.add_fail("CP key station count", f"Expected >= 10, got {len(CP_KEY_STATIONS)}")
        
        # Test 3: GTFS database
        manager = CPGTFSManager()
        if manager.db_path.exists():
            results.add_pass("CP GTFS database exists")
            
            # Test database contents
            conn = manager.get_db_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM stops")
                stops_count = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM routes")
                routes_count = cursor.fetchone()[0]
                conn.close()
                
                if stops_count > 400:
                    results.add_pass(f"CP GTFS has {stops_count} stops")
                else:
                    results.add_warning("CP GTFS stops", f"Only {stops_count} stops found")
                
                if routes_count > 100:
                    results.add_pass(f"CP GTFS has {routes_count} routes")
                else:
                    results.add_warning("CP GTFS routes", f"Only {routes_count} routes found")
        else:
            results.add_warning("CP GTFS", "Database not found, initializing...")
            init_result = initialize_cp_gtfs.invoke({"force_refresh": False})
            if "successful" in init_result.lower() or "up-to-date" in init_result.lower():
                results.add_pass("CP GTFS initialized successfully")
            else:
                results.add_fail("CP GTFS initialization", "Failed to initialize")
        
        # Test 4: Get train status (real-time API)
        status = get_train_status.invoke({})
        if status and "CP Trains" in status:
            results.add_pass("get_train_status returns valid response")
            
            # Check for AML filtering
            if "AML" in status:
                results.add_pass("Train status filters to AML region")
        else:
            results.add_fail("get_train_status", "Invalid response format")
        
        # Test 5: Search stations
        stations = search_cp_stations.invoke({"query": "Oriente"})
        if stations and "Oriente" in stations:
            results.add_pass("search_cp_stations finds Lisboa Oriente")
        else:
            results.add_fail("search_cp_stations", "Could not find Oriente station")
        
        # Test 6: Get train schedule (GTFS-based)
        schedule = get_train_schedule.invoke({"station_name": "Lisboa", "limit": 5})
        if schedule and "Departures" in schedule:
            results.add_pass("get_train_schedule returns GTFS-based schedule")
        else:
            results.add_fail("get_train_schedule", "Invalid schedule format")
        
    except ImportError as e:
        results.add_fail("CP API Import", str(e))
    except Exception as e:
        results.add_fail("CP API", str(e))


def test_ipma_api(results: TestResults) -> None:
    """Tests IPMA Weather API functionality."""
    print("\n" + "-" * 70)
    print("\033[1m🌤️ TEST: IPMA Weather API\033[0m")
    print("-" * 70)
    
    try:
        from tools.ipma_api import (
            get_weather_forecast,
            get_weather_warnings,
            get_current_weather_summary,
        )
        from config import Config
        
        # Test 1: Lisbon ID configured
        if Config.LISBON_GLOBAL_ID == 1110600:
            results.add_pass("Lisbon Global ID configured correctly (1110600)")
        else:
            results.add_fail("Lisbon Global ID", f"Expected 1110600, got {Config.LISBON_GLOBAL_ID}")
        
        # Test 2: Get weather forecast
        forecast = get_weather_forecast.invoke({"days": 3})
        if forecast and "Forecast" in forecast and "Lisbon" in forecast:
            results.add_pass("get_weather_forecast returns 3-day forecast")
        else:
            results.add_fail("get_weather_forecast", "Invalid forecast format")
        
        # Test 3: Get weather warnings
        warnings = get_weather_warnings.invoke({"area": "LSB"})
        if warnings and ("Warning" in warnings or "No active" in warnings):
            results.add_pass("get_weather_warnings returns Lisbon warnings")
        else:
            results.add_fail("get_weather_warnings", "Invalid warnings format")
        
        # Test 4: Get current weather summary
        summary = get_current_weather_summary.invoke({})
        if summary and "Summary" in summary and "Temperature" in summary:
            results.add_pass("get_current_weather_summary returns valid summary")
        else:
            results.add_fail("get_current_weather_summary", "Invalid summary format")
        
    except ImportError as e:
        results.add_fail("IPMA API Import", str(e))
    except Exception as e:
        results.add_fail("IPMA API", str(e))


def test_transport_integration(results: TestResults) -> None:
    """Tests multi-modal transport integration."""
    print("\n" + "-" * 70)
    print("\033[1m🗺️ TEST: Multi-Modal Transport Integration\033[0m")
    print("-" * 70)
    
    try:
        from tools.transport_api import (
            get_transport_summary,
            get_route_between_stations
        )
        
        # Test 1: Get transport summary (all modes)
        summary = get_transport_summary.invoke({})
        if summary:
            checks = [
                ("METRO" in summary, "Metro included"),
                ("CARRIS" in summary, "Carris included"),
                ("CP" in summary or "TRAINS" in summary, "CP Trains included"),
            ]
            
            for check, name in checks:
                if check:
                    results.add_pass(f"Transport summary: {name}")
                else:
                    results.add_fail(f"Transport summary: {name}", "Section missing")
        else:
            results.add_fail("get_transport_summary", "Empty response")
        
        # Test 2: Route between Metro stations (same line)
        route1 = get_route_between_stations.invoke({
            "origin": "Baixa-Chiado",
            "destination": "Cais do Sodré"
        })
        if route1 and "Route" in route1:
            results.add_pass("Route calculation: Metro same line")
        else:
            results.add_fail("Route calculation", "Failed for Metro same line")
        
        # Test 3: Route between Metro stations (transfer required)
        route2 = get_route_between_stations.invoke({
            "origin": "Aeroporto",
            "destination": "Baixa-Chiado"
        })
        if route2 and ("Transfer" in route2 or "transfer" in route2):
            results.add_pass("Route calculation: Metro with transfer")
        else:
            results.add_warning("Route calculation", "Transfer not detected (may be valid)")
        
        # Test 4: Route from landmark to Metro
        route3 = get_route_between_stations.invoke({
            "origin": "Colombo",
            "destination": "Oriente"
        })
        if route3 and "Route" in route3:
            results.add_pass("Route calculation: Landmark to Metro")
        else:
            results.add_fail("Route calculation", "Failed for Landmark to Metro")
        
        # Test 5: Route involving train station
        route4 = get_route_between_stations.invoke({
            "origin": "Rossio",
            "destination": "Sintra"
        })
        if route4:
            if "train" in route4.lower() or "CP" in route4 or "Linha" in route4:
                results.add_pass("Route calculation: Train route suggested")
            else:
                results.add_warning("Route calculation", "Train option not shown for Rossio-Sintra")
        else:
            results.add_fail("Route calculation", "Failed for train route")
        
    except ImportError as e:
        results.add_fail("Transport Integration Import", str(e))
    except Exception as e:
        results.add_fail("Transport Integration", str(e))


def test_data_consistency(results: TestResults) -> None:
    """Tests data consistency across APIs."""
    print("\n" + "-" * 70)
    print("\033[1m🔍 TEST: Data Consistency Checks\033[0m")
    print("-" * 70)
    
    try:
        from tools.metrolisboa_api import METRO_STATIONS, METRO_LINES, LISBON_LANDMARKS
        from tools.cp_api import CP_LINES, CP_KEY_STATIONS
        
        # Test 1: Metro stations have valid line references
        # METRO_STATIONS format: {"station_name": ["line1", "line2"], ...}
        valid_lines = set(METRO_LINES.keys())
        invalid_refs = []
        for station, lines in METRO_STATIONS.items():
            # Lines can be a list directly or a dict with 'lines' key
            station_lines = lines if isinstance(lines, list) else lines.get('lines', [])
            for line in station_lines:
                if line not in valid_lines:
                    invalid_refs.append(f"{station}: {line}")
        
        if not invalid_refs:
            results.add_pass("Metro stations have valid line references")
        else:
            results.add_fail("Metro line references", f"Invalid: {invalid_refs[:3]}")
        
        # Test 2: CP stations have valid line references
        valid_cp_lines = set(CP_LINES.keys())
        invalid_cp_refs = []
        for station, info in CP_KEY_STATIONS.items():
            for line in info.get('lines', []):
                if line not in valid_cp_lines:
                    invalid_cp_refs.append(f"{station}: {line}")
        
        if not invalid_cp_refs:
            results.add_pass("CP stations have valid line references")
        else:
            results.add_fail("CP line references", f"Invalid: {invalid_cp_refs[:3]}")
        
        # Test 3: Landmarks have metro references that exist
        invalid_landmarks = []
        for landmark, info in LISBON_LANDMARKS.items():
            metro = info.get('metro', '')
            if metro and metro.lower() not in [s.lower() for s in METRO_STATIONS.keys()]:
                invalid_landmarks.append(f"{landmark}: {metro}")
        
        if not invalid_landmarks:
            results.add_pass("Landmarks have valid metro references")
        else:
            results.add_warning("Landmark metro references", f"Check: {invalid_landmarks[:3]}")
        
        # Test 4: All metro lines have stations
        lines_with_stations = set()
        for station, lines in METRO_STATIONS.items():
            # Lines can be a list directly or a dict with 'lines' key
            station_lines = lines if isinstance(lines, list) else lines.get('lines', [])
            for line in station_lines:
                lines_with_stations.add(line)
        
        missing_lines = valid_lines - lines_with_stations
        if not missing_lines:
            results.add_pass("All metro lines have stations assigned")
        else:
            results.add_fail("Metro line coverage", f"Lines without stations: {missing_lines}")
        
    except ImportError as e:
        results.add_fail("Data Consistency Import", str(e))
    except Exception as e:
        results.add_fail("Data Consistency", str(e))


# ==========================================================================
# Main Execution
# ==========================================================================

def main():
    """Main test execution."""
    print("\n" + "=" * 70)
    print("\033[1m🧪 COMPREHENSIVE LISBON TRANSPORT API TEST SUITE\033[0m")
    print("=" * 70)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print("=" * 70)
    
    results = TestResults()
    
    # Run all test suites
    test_metro_api(results)
    test_carris_api(results)
    test_carris_metropolitana_api(results)
    test_cp_api(results)
    test_ipma_api(results)
    test_transport_integration(results)
    test_data_consistency(results)
    
    # Print summary
    print(results.summary())
    
    # Return exit code
    return 0 if results.failed == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
