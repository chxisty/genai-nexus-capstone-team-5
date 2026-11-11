import unittest

from tools.database_tools import (
    search_stations,
    search_trains,
    get_train_details,
    get_train_route,
)


class TestDatabaseTools(unittest.TestCase):

    def test_station_search(self):
        result = search_stations("Delhi")
        self.assertGreater(result["count"], 0)
        self.assertIn("matches", result)

    def test_train_search(self):
        result = search_trains("LTT", "GWL", "2026-11-15")
        self.assertEqual(result["count"], 6)
        self.assertEqual(result["source"]["code"], "LTT")
        self.assertEqual(result["destination"]["code"], "GWL")

    def test_train_details(self):
        result = get_train_details("01101")
        self.assertEqual(result["number"], "01101")
        self.assertIn("3A", result["classes"])
        self.assertIn("SL", result["classes"])

    def test_train_route(self):
        result = get_train_route("01101")
        self.assertGreater(result["count"], 0)
        self.assertEqual(result["stops"][0]["station_code"], "LTT")
        self.assertEqual(result["stops"][-1]["station_code"], "GWL")

    def test_unknown_train(self):
        result = get_train_details("99999")
        self.assertIn("error", result)

    def test_unknown_station(self):
        result = search_trains(
            "INVALID_STATION",
            "GWL",
            "2026-11-15"
        )
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
