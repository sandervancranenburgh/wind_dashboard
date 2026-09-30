from __future__ import annotations

import copy
import unittest
from datetime import date

from next_day_wind_model.site_registry import (
    display_name,
    enabled_sites,
    load_site_registry,
    registry_from_mapping,
)


class SiteRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        load_site_registry.cache_clear()

    def test_production_site_contract(self) -> None:
        registry = load_site_registry()
        self.assertEqual(registry.site_ids, ("valkenburgsemeer", "oostvoorne"))
        self.assertEqual(registry.default_public_site_id, "valkenburgsemeer")
        self.assertEqual(display_name("valkenburgsemeer"), "Valkenburgse Meer")
        self.assertEqual(display_name("oostvoorne"), "Oostvoornse Meer")
        self.assertEqual(registry.canonical_site_id("Oostvoornse meer"), "oostvoorne")
        self.assertEqual(registry.canonical_site_id("Oostvoornse Meer"), "oostvoorne")
        self.assertEqual(
            registry.rider_spot_values(),
            ("Valkenburgse meer", "Oostvoornse meer"),
        )

    def test_current_operational_enablement_is_preserved(self) -> None:
        self.assertEqual(
            tuple(site.site_id for site in enabled_sites("windsurfice")),
            ("valkenburgsemeer", "oostvoorne"),
        )
        self.assertEqual(
            tuple(site.site_id for site in enabled_sites("ecmwf")),
            ("valkenburgsemeer", "oostvoorne"),
        )
        self.assertEqual(
            tuple(site.site_id for site in enabled_sites("knmi_p1")),
            ("valkenburgsemeer", "oostvoorne"),
        )
        self.assertEqual(
            tuple(site.site_id for site in enabled_sites("superlocal_model")),
            ("valkenburgsemeer",),
        )
        self.assertEqual(
            load_site_registry().site("oostvoorne").superlocal_model.production_eligible_after,
            date(2027, 4, 1),
        )

    def test_source_specific_coordinates_are_retained(self) -> None:
        site = load_site_registry().site("valkenburgsemeer")
        self.assertEqual(
            (site.windsurfice.forecast_latitude, site.windsurfice.forecast_longitude),
            (52.1603, 4.44197),
        )
        self.assertEqual((site.knmi_p1.latitude, site.knmi_p1.longitude), (52.168, 4.437))
        oostvoorne = load_site_registry().site("oostvoorne")
        self.assertEqual(
            (oostvoorne.knmi_p1.latitude, oostvoorne.knmi_p1.longitude),
            (51.928597, 4.074097),
        )
        self.assertEqual(
            (oostvoorne.windsurfice.forecast_latitude, oostvoorne.windsurfice.forecast_longitude),
            (51.9278, 4.05502),
        )

    def test_duplicate_site_id_is_rejected(self) -> None:
        registry = load_site_registry()
        raw = {
            "schema_version": 1,
            "sites": [],
        }
        config_path = load_site_registry.__wrapped__.__defaults__[0]
        import json

        value = json.loads(config_path.read_text(encoding="utf-8"))
        duplicate = copy.deepcopy(value["sites"][0])
        value["sites"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "site_id values must be unique"):
            registry_from_mapping(value)

    def test_cross_site_alias_collision_is_rejected(self) -> None:
        config_path = load_site_registry.__wrapped__.__defaults__[0]
        import json

        value = json.loads(config_path.read_text(encoding="utf-8"))
        value["sites"][1]["aliases"].append("Valkenburgse Meer")
        with self.assertRaisesRegex(ValueError, "shared"):
            registry_from_mapping(value)

    def test_invalid_coordinate_is_rejected(self) -> None:
        config_path = load_site_registry.__wrapped__.__defaults__[0]
        import json

        value = json.loads(config_path.read_text(encoding="utf-8"))
        value["sites"][0]["ecmwf"]["latitude"] = 91
        with self.assertRaisesRegex(ValueError, "latitude"):
            registry_from_mapping(value)


if __name__ == "__main__":
    unittest.main()
