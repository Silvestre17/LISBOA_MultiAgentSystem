# Carris Urban API Tools Documentation

**Module**: `tools/carris_api.py`  
**Last Updated**: January 2026

---

## Overview

This module provides tools for accessing **Carris urban bus and tram data** in Lisbon city center. It combines:

1. **GTFS Static Data**: Routes, stops, schedules, and trip information
2. **GTFS-RT Real-Time Feed**: Live vehicle positions via Protocol Buffers

**Operator**: Carris (Lisbon city, NOT Carris Metropolitana)  
**Vehicles**: ~400 urban buses + ~25 historic trams (28E, 15E, etc.)

---

## Data Sources

### GTFS Static (ZIP Archive)
- **URL**: https://gateway.carris.pt/gateway/gtfs/api/v2.8/GTFS
- **Format**: ZIP file containing CSV files (routes.txt, stops.txt, trips.txt, stop_times.txt, etc.)
- **Update**: Daily (checked via HTTP headers)
- **Storage**: SQLite database at `data/carris/carris.db`
- **Size**: ~34MB ZIP → ~150MB SQLite with indexes

### GTFS-RT Real-Time (Protocol Buffers)
- **URL**: https://gateway.carris.pt/gateway/gtfs/api/v2.8/GTFS/realtime/vehiclepositions
- **Format**: Protocol Buffers (requires `gtfs-realtime-bindings` library)
- **Update**: Real-time (cached for 30 seconds)
- **Data**: Vehicle positions, trip associations, next stop, license plates

---

## Database Schema

The SQLite database contains the following tables:

| Table | Records | Description |
|-------|---------|-------------|
| `routes` | ~175 | Bus and tram lines (28E, 15E, 732, etc.) |
| `stops` | ~2,335 | Bus/tram stops with GPS coordinates |
| `trips` | ~78,000 | Individual vehicle trips |
| `stop_times` | ~2.1M | Scheduled arrivals/departures at each stop |
| `calendar` | - | Service days (weekday, Saturday, Sunday) |
| `calendar_dates` | - | Exceptions (holidays, special days) |
| `shapes` | - | Route geometry for mapping |

### Key Indexes
- `idx_stops_name` - Fast stop name search
- `idx_routes_short` - Route short name lookup
- `idx_stop_times_stop_dep` - Arrivals by stop and time
- `idx_stop_times_trip_seq` - Trip sequence for ETA calculation

---

## Available Tools (7)

### 1. `carris_get_stops(query, limit)`

Search for Carris stops by name.

**Parameters**:
- `query` (str): Stop name to search (partial match)
- `limit` (int): Maximum results (default: 20)

**Returns**: List of stops with ID, name, GPS coordinates

**Example**:
```python
carris_get_stops.invoke({"query": "Rossio", "limit": 5})
```

**Output**:
```
Paragens Carris (pesquisa: 'Rossio')
=============================================

PARAGEM: Rossio
   ID: 908 | Código: 908
   GPS: 38.71331, -9.13962
...
```

---

### 2. `carris_get_routes(route_type, route_id, limit)`

List Carris bus and tram routes.

**Parameters**:
- `route_type` (str): Filter by "bus", "tram", "elétrico", "autocarro" (optional)
- `route_id` (str): Search for specific route (optional)
- `limit` (int): Maximum results (default: 50)

**Returns**: Routes grouped by trams and buses

**Example**:
```python
carris_get_routes.invoke({"route_type": "tram"})
```

---

### 3. `carris_get_arrivals(stop_id, limit)` ⭐ PRIMARY

**Real-time arrivals at a stop**, combining GTFS schedule with GTFS-RT vehicle positions.

**This is the BEST tool for "when is the next bus/tram at stop X".**

**Parameters**:
- `stop_id` (str): Stop ID from `carris_get_stops`
- `limit` (int): Maximum arrivals (default: 10)

**Returns**: Arrivals with:
- Scheduled vs estimated times
- Delay information
- Vehicle ID and license plate
- Stops remaining

**Example**:
```python
carris_get_arrivals.invoke({"stop_id": "13810", "limit": 5})
```

**Output**:
```
Próximas Chegadas: Castelo
   ID: 13810 | Atualizado: 14:21
=======================================================

[TEMPO REAL] Autocarro 737 -> Pç. Figueira - Castelo
   Hora: 14:29 (atrasado 1 min)
   Faltam 5 paragens
   Veículo: 2971 | Matrícula: 29-VM-98

[HORÁRIO] Autocarro 737 -> Pç. Figueira - Castelo
   Hora: 14:35
...
```

---

### 4. `carris_get_stop_schedule(stop_id, limit)`

Static schedule for a stop (no real-time data).

**Parameters**:
- `stop_id` (str): Stop ID
- `limit` (int): Maximum departures (default: 15)

**Returns**: Scheduled departures without real-time adjustments

---

### 5. `carris_find_routes_between(origin, destination, search_radius_km)`

Find direct routes between two locations.

**Parameters**:
- `origin` (str): Origin location or stop name
- `destination` (str): Destination location or stop name
- `search_radius_km` (float): Search radius (default: 0.4km, auto-expands)

**Returns**: Available routes with next departure times

**Example**:
```python
carris_find_routes_between.invoke({
    "origin": "Martim Moniz",
    "destination": "Belém"
})
```

---

### 6. `carris_get_realtime_vehicles(route_id, vehicle_type)`

Get real-time positions of all Carris vehicles.

**Parameters**:
- `route_id` (str): Filter by route (e.g., "28E") - optional
- `vehicle_type` (str): Filter by "tram" or "bus" - optional

**Returns**: Live GPS positions with:
- Route and destination
- Current status (in transit, stopped)
- Next stop
- License plate

**Example**:
```python
carris_get_realtime_vehicles.invoke({"route_id": "28E"})
```

**Output**:
```
Veículos Carris em Tempo Real
=======================================================
Dados de: 14:22:16

ELÉTRICOS
----------------------------------------
28E -> Martim Moniz [Em trânsito]
   GPS: 38.71430, -9.16937 | Próxima paragem: Campo Ourique
   Matrícula: AG-26-NH
...
```

---

### 7. `carris_vehicle_eta(route_short_name, stop_name)` ⭐ PRIMARY

Calculate estimated arrival time for a specific route at a stop.

**This is the BEST tool for "when will bus/tram X arrive at stop Y".**

**Parameters**:
- `route_short_name` (str): Route number (e.g., "28E", "732")
- `stop_name` (str): Stop name (partial match)

**Returns**: ETAs for all vehicles of that route approaching the stop

**Example**:
```python
carris_vehicle_eta.invoke({
    "route_short_name": "28E",
    "stop_name": "Graça"
})
```

---

## GTFS-RT Data Structure

Each vehicle in the GTFS-RT feed contains:

```json
{
    "entity_id": "4637",
    "feed_timestamp": 1769004239,
    "trip_id": "8140_20260101_174_0_24",
    "route_id": "174_0",
    "direction_id": 0,
    "latitude": 38.7656,
    "longitude": -9.1573,
    "vehicle_id": "4637",
    "license_plate": "29-VM-98",
    "current_status": "IN_TRANSIT_TO",
    "stop_id": "5902",
    "timestamp": 1769004230
}
```

### Status Codes
- `IN_TRANSIT_TO` (2): Vehicle is traveling toward the stop
- `STOPPED_AT` (1): Vehicle is currently at the stop
- `INCOMING_AT` (0): Vehicle is approaching the stop

---

## Data Joins

The GTFS-RT data can be joined with static GTFS data:

1. **route_id** → `routes.route_id` → Get route name (e.g., "28E")
2. **trip_id** → `trips.trip_id` → Get trip headsign, direction
3. **stop_id** → `stops.stop_id` → Get stop name, GPS coordinates
4. **trip_id + stop_id** → `stop_times` → Calculate ETA based on schedule

---

## Tram Lines Reference

| Route | Name | Description |
|-------|------|-------------|
| **12E** | Martim Moniz - Pç. Luis Camões | Historic tram through Alfama |
| **15E** | Pç. Figueira - Algés | Riverside tram to Belém |
| **18E** | Cais Sodré - Cemitério Ajuda | Western suburbs |
| **24E** | Pç. Luis Camões - Campolide | Bairro Alto to Campolide |
| **25E** | Pç Figueira - Campo Ourique | Through historic center |
| **28E** | Martim Moniz - Pç. Luis Camões | Famous tourist tram (Graça, Alfama, Chiado) |

---

## Dependencies

Required libraries in `requirements.txt`:

```
protobuf>=3.20.0
gtfs-realtime-bindings>=0.0.0
requests
langchain-core
```

---

## Cache Strategy

- **GTFS Static**: Checked daily via HTTP headers (Content-Disposition filename contains date)
- **GTFS-RT**: Cached for 30 seconds to avoid excessive API calls
- **Database**: Persisted in SQLite for fast queries

---

## Error Handling

The tools handle various error conditions:

1. **Network timeout**: Uses cached data if available
2. **Database unavailable**: Attempts auto-download of GTFS
3. **No active services**: Returns appropriate message (e.g., holidays)
4. **Stop not found**: Suggests using `carris_get_stops` first

---

## Example Workflows

### "When is the next bus to Castelo de São Jorge?"

```python
# Step 1: Find the stop
stops = carris_get_stops.invoke({"query": "Castelo", "limit": 3})
# Returns stop_id: 13810

# Step 2: Get real-time arrivals
arrivals = carris_get_arrivals.invoke({"stop_id": "13810", "limit": 5})
# Returns arrivals with real-time delays
```

### "Where are all the tram 28E right now?"

```python
vehicles = carris_get_realtime_vehicles.invoke({"route_id": "28E"})
# Returns GPS positions of all 28E trams currently in service
```

### "How do I get from Rossio to Belém by tram?"

```python
routes = carris_find_routes_between.invoke({
    "origin": "Rossio",
    "destination": "Belém"
})
# Returns: Tram 15E with next departure times
```
