"""Real SQLite-backed railway functions for the team project."""

import sqlite3
from pathlib import Path
from contextlib import closing

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "rail_timetable.db"

STATIC_NOTE = (
    "Based on the static railway timetable database. "
    "This does not provide live running status or seat availability."
)


def get_connection():
    """Open the railway database in read-only mode."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Railway database not found: {DB_PATH}")

    return sqlite3.connect(f"{DB_PATH.as_uri()}?mode=ro", uri=True)


def search_stations(query: str) -> dict:
    """Search stations by name or station code."""
    query = query.strip()

    with closing(get_connection()) as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            """
            SELECT code, name, state, zone
            FROM stations
            WHERE code LIKE ? OR name LIKE ?
            ORDER BY
                CASE WHEN UPPER(code) = UPPER(?) THEN 0 ELSE 1 END,
                name
            LIMIT 20
            """,
            (f"%{query}%", f"%{query}%", query),
        ).fetchall()

    return {
        "query": query,
        "count": len(rows),
        "matches": [dict(row) for row in rows],
        "source": "rail_timetable.db",
    }


def _resolve_station(conn, value: str):
    """Resolve a station by exact code, exact name, or partial match."""
    value = value.strip()

    row = conn.execute(
        "SELECT code, name FROM stations WHERE UPPER(code) = UPPER(?)",
        (value,),
    ).fetchone()

    if row:
        return dict(row), None

    rows = conn.execute(
        """
        SELECT code, name
        FROM stations
        WHERE LOWER(name) = LOWER(?)
        ORDER BY name
        LIMIT 10
        """,
        (value,),
    ).fetchall()

    if len(rows) == 1:
        return dict(rows[0]), None

    if len(rows) > 1:
        return None, {
            "error": f"Multiple stations match '{value}'. Use a station code.",
            "matches": [dict(row) for row in rows],
        }

    rows = conn.execute(
        """
        SELECT code, name
        FROM stations
        WHERE LOWER(name) LIKE LOWER(?) OR UPPER(code) LIKE UPPER(?)
        ORDER BY name
        LIMIT 10
        """,
        (f"%{value}%", f"%{value}%"),
    ).fetchall()

    if len(rows) == 1:
        return dict(rows[0]), None

    if rows:
        return None, {
            "error": f"Station '{value}' is ambiguous. Use a station code.",
            "matches": [dict(row) for row in rows],
        }

    return None, {"error": f"No station found for '{value}'."}


def search_trains(source: str, destination: str, date: str) -> dict:
    """Find trains travelling between two stations using scheduled stops."""
    with closing(get_connection()) as conn:
        conn.row_factory = sqlite3.Row

        source_station, source_error = _resolve_station(conn, source)
        if source_error:
            return {"source": source, "destination": destination, **source_error}

        destination_station, destination_error = _resolve_station(
            conn, destination
        )
        if destination_error:
            return {
                "source": source,
                "destination": destination,
                **destination_error,
            }

        if source_station["code"] == destination_station["code"]:
            return {"error": "Source and destination cannot be the same station."}

        rows = conn.execute(
            """
            SELECT
                t.number AS train_number,
                t.name,
                t.type_code,
                tt.meaning AS train_type,
                src.code AS source_code,
                src.name AS source_name,
                dst.code AS destination_code,
                dst.name AS destination_name,
                s1.departure,
                s2.arrival,
                s1.day AS source_day,
                s2.day AS destination_day
            FROM schedules s1
            JOIN schedules s2
                ON s1.train_number = s2.train_number
                AND s1.stop_sequence < s2.stop_sequence
            JOIN trains t ON t.number = s1.train_number
            JOIN stations src ON src.code = s1.station_code
            JOIN stations dst ON dst.code = s2.station_code
            LEFT JOIN train_types tt ON tt.type_code = t.type_code
            WHERE s1.station_code = ?
              AND s2.station_code = ?
            ORDER BY t.number
            LIMIT 20
            """,
            (source_station["code"], destination_station["code"]),
        ).fetchall()

    return {
        "source": source_station,
        "destination": destination_station,
        "requested_date": date,
        "count": len(rows),
        "trains": [dict(row) for row in rows],
        "note": STATIC_NOTE + " The requested date is not used to verify service days.",
    }


def get_train_details(train_number: str) -> dict:
    """Retrieve train details and available coach classes."""
    train_number = train_number.strip()

    with closing(get_connection()) as conn:
        conn.row_factory = sqlite3.Row

        row = conn.execute(
            """
            SELECT
                t.*,
                tt.meaning AS train_type,
                src.name AS source_name,
                dst.name AS destination_name
            FROM trains t
            LEFT JOIN train_types tt ON tt.type_code = t.type_code
            LEFT JOIN stations src ON src.code = t.from_station_code
            LEFT JOIN stations dst ON dst.code = t.to_station_code
            WHERE t.number = ?
            """,
            (train_number,),
        ).fetchone()

    if not row:
        return {"error": f"Train '{train_number}' not found."}

    details = dict(row)

    class_columns = {
        "1A": "has_1a",
        "2A": "has_2a",
        "3A": "has_3a",
        "SL": "has_sl",
        "CC": "has_cc",
        "FC": "has_fc",
    }

    details["classes"] = [
        coach_class
        for coach_class, column in class_columns.items()
        if details[column]
    ]

    details["note"] = STATIC_NOTE
    return details


def get_train_route(train_number: str) -> dict:
    """Retrieve the scheduled stops for a train."""
    train_number = train_number.strip()

    with closing(get_connection()) as conn:
        conn.row_factory = sqlite3.Row

        train = conn.execute(
            "SELECT number, name FROM trains WHERE number = ?",
            (train_number,),
        ).fetchone()

        if not train:
            return {"error": f"Train '{train_number}' not found."}

        rows = conn.execute(
            """
            SELECT
                s.stop_sequence,
                s.station_code,
                st.name AS station_name,
                st.state,
                s.day,
                s.arrival,
                s.departure,
                s.halt_min,
                s.is_halt
            FROM schedules s
            LEFT JOIN stations st ON st.code = s.station_code
            WHERE s.train_number = ?
            ORDER BY s.stop_sequence
            """,
            (train_number,),
        ).fetchall()

    return {
        "train_number": train["number"],
        "train_name": train["name"],
        "count": len(rows),
        "stops": [dict(row) for row in rows],
        "note": STATIC_NOTE,
    }

def register_database_tools(registry):
    """Register real database-backed railway tools."""
    registry.register("station_search", search_stations)
    registry.register("train_search", search_trains)
    registry.register("train_details", get_train_details)
    registry.register("route_lookup", get_train_route)
