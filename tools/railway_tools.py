"""
railway_tools.py — Mock railway functions for demo and testing.

IMPORTANT:  These return **hard-coded sample data** so the module can be
demonstrated and tested independently.  Krishna (Member 2) will later
provide real database-backed functions.  To swap in real implementations,
simply register them on the same ToolRegistry instance:

    registry.register("station_search", krishna_db.search_stations)

No changes to the registry or schemas are needed.
"""


# -----------------------------------------------------------------------
# Mock implementations  (clearly labelled — not real data)
# -----------------------------------------------------------------------

def mock_station_search(query: str) -> dict:
    """[MOCK] Search stations by partial name."""
    sample_stations = [
        {"code": "NDLS", "name": "New Delhi"},
        {"code": "BCT",  "name": "Mumbai Central"},
        {"code": "MAS",  "name": "Chennai Central"},
        {"code": "HWH",  "name": "Howrah Junction"},
        {"code": "SBC",  "name": "KSR Bengaluru"},
    ]
    query_lower = query.lower()
    matches = [
        s for s in sample_stations
        if query_lower in s["name"].lower() or query_lower in s["code"].lower()
    ]
    return {"query": query, "matches": matches}


def mock_train_search(source: str, destination: str, date: str) -> dict:
    """[MOCK] Search trains between two stations on a date."""
    return {
        "source": source,
        "destination": destination,
        "date": date,
        "trains": [
            {
                "train_number": "12345",
                "name": "Mock Express",
                "departure": "06:00",
                "arrival": "14:00",
                "duration": "8h 00m",
            },
            {
                "train_number": "67890",
                "name": "Mock Superfast",
                "departure": "20:00",
                "arrival": "04:00+1",
                "duration": "8h 00m",
            },
        ],
        "_note": "MOCK DATA — not real schedules.",
    }


def mock_train_details(train_number: str) -> dict:
    """[MOCK] Get details for a train by number."""
    return {
        "train_number": train_number,
        "name": "Mock Express",
        "type": "Superfast",
        "runs_on": ["Mon", "Wed", "Fri"],
        "classes": ["SL", "3A", "2A", "1A"],
        "_note": "MOCK DATA — not real train information.",
    }


def mock_route_lookup(train_number: str) -> dict:
    """[MOCK] Look up the route for a train by number."""
    return {
        "train_number": train_number,
        "stops": [
            {"station": "New Delhi",       "arrival": "--",    "departure": "06:00"},
            {"station": "Agra Cantt",      "arrival": "08:10", "departure": "08:15"},
            {"station": "Jhansi Junction", "arrival": "10:30", "departure": "10:35"},
            {"station": "Bhopal Junction", "arrival": "13:50", "departure": "14:00"},
        ],
        "_note": "MOCK DATA — not a real route.",
    }


# -----------------------------------------------------------------------
# Convenience: register all mocks on a registry instance
# -----------------------------------------------------------------------

def register_railway_tools(registry) -> None:
    """Register the four mock railway tools on *registry*."""
    registry.register("station_search", mock_station_search)
    registry.register("train_search",   mock_train_search)
    registry.register("train_details",  mock_train_details)
    registry.register("route_lookup",   mock_route_lookup)
