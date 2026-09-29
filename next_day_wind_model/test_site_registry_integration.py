from __future__ import annotations

import unittest
from pathlib import Path

import db_store
import source_fetch
from next_day_wind_model.shadow_config import load_shadow_config
from scripts.knmi_extract_latest_to_db import DEFAULT_SITE_POINTS


class SiteRegistryIntegrationTests(unittest.TestCase):
    def test_windsurfice_configs_preserve_verified_source_values(self) -> None:
        self.assertEqual(tuple(source_fetch.SITES), ("valkenburgsemeer", "oostvoorne"))
        self.assertEqual(
            source_fetch.SITES["valkenburgsemeer"].observation_params["Site"],
            "windsurfice-v25-node7",
        )
        self.assertEqual(
            source_fetch.SITES["oostvoorne"].forecast_payload,
            {"lat": "51.9278", "lon": "4.05502"},
        )

    def test_source_fetch_requires_explicit_selection(self) -> None:
        with self.assertRaises(SystemExit):
            source_fetch.parse_args(["data"])
        selected = source_fetch.parse_args(
            ["data", "--site", "oostvoorne", "--site", "valkenburgsemeer"]
        )
        self.assertEqual(selected.sites, ["oostvoorne", "valkenburgsemeer"])
        self.assertFalse(selected.all_sites)
        all_sites = source_fetch.parse_args(["data", "--all-sites"])
        self.assertTrue(all_sites.all_sites)

    def test_database_spot_aliases_resolve_without_row_rewrite(self) -> None:
        self.assertEqual(db_store.SPOT_TO_SITE["Oostvoornse meer"], "oostvoorne")
        self.assertEqual(db_store.SPOT_TO_SITE["Oostvoornse Meer"], "oostvoorne")
        self.assertEqual(db_store.SPOT_TO_SITE["Valkenburgse meer"], "valkenburgsemeer")
        self.assertEqual(db_store.SPOT_TO_SITE["Valkenburgse Meer"], "valkenburgsemeer")

    def test_knmi_operational_enablement_remains_valkenburg_only(self) -> None:
        self.assertEqual(tuple(DEFAULT_SITE_POINTS), ("valkenburgsemeer",))

    def test_production_ecmwf_config_resolves_both_registry_sites(self) -> None:
        config = load_shadow_config(Path("config/ecmwf_production.json"))
        self.assertEqual(
            tuple(site.site_id for site in config.sites),
            ("valkenburgsemeer", "oostvoorne"),
        )
        self.assertEqual(
            (config.site("oostvoorne").latitude, config.site("oostvoorne").longitude),
            (51.9278, 4.05502),
        )


if __name__ == "__main__":
    unittest.main()
