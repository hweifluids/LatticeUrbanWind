from __future__ import annotations

from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = REPO_ROOT / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from deck_io import parse_deck_text
from terr_voxel_config import resolve_terrain_voxel_config


class TerrainVoxelConfigTests(unittest.TestCase):
    def test_deck_values_override_defaults_when_cli_omits_them(self) -> None:
        deck = parse_deck_text(
            """
            // Domain
            terr_voxel_height_field = BUILD_H
            terr_voxel_ignore_under = 4.0
            terr_voxel_approach = kriging
            terr_voxel_grid_resolution = 20.0
            terr_voxel_idw_sigma = 0.25
            terr_voxel_idw_power = 1.5
            terr_voxel_idw_neighbors = 6
            """
        )

        config, sources = resolve_terrain_voxel_config(deck)

        self.assertEqual(config.height_field, "BUILD_H")
        self.assertEqual(config.ignore_under, 4.0)
        self.assertEqual(config.approach, "kriging")
        self.assertEqual(config.grid_resolution, 20.0)
        self.assertEqual(config.idw_sigma, 0.25)
        self.assertEqual(config.idw_power, 1.5)
        self.assertEqual(config.idw_neighbors, 6)
        self.assertEqual(sources["approach"], "deck")
        self.assertEqual(sources["grid_resolution"], "deck")

    def test_cli_values_override_deck_values(self) -> None:
        deck = parse_deck_text(
            """
            // Domain
            terr_voxel_height_field = DECK_HEIGHT
            terr_voxel_ignore_under = 2.0
            terr_voxel_approach = idw
            terr_voxel_grid_resolution = 50.0
            terr_voxel_idw_sigma = 1.0
            terr_voxel_idw_power = 2.0
            terr_voxel_idw_neighbors = 12
            """
        )

        config, sources = resolve_terrain_voxel_config(
            deck,
            cli_overrides={
                "height_field": "CLI_HEIGHT",
                "ignore_under": 5.0,
                "approach": "kriging",
                "grid_resolution": 15.0,
                "idw_sigma": 0.1,
                "idw_power": 1.2,
                "idw_neighbors": 4,
            },
        )

        self.assertEqual(config.height_field, "CLI_HEIGHT")
        self.assertEqual(config.ignore_under, 5.0)
        self.assertEqual(config.approach, "kriging")
        self.assertEqual(config.grid_resolution, 15.0)
        self.assertEqual(config.idw_sigma, 0.1)
        self.assertEqual(config.idw_power, 1.2)
        self.assertEqual(config.idw_neighbors, 4)
        self.assertTrue(all(source == "cli" for source in sources.values()))

    def test_gpu_kriging_is_an_allowed_choice(self) -> None:
        deck = parse_deck_text(
            """
            // Domain
            terr_voxel_approach = kriging_gpu
            """
        )

        config, sources = resolve_terrain_voxel_config(deck)

        self.assertEqual(config.approach, "kriging_gpu")
        self.assertEqual(sources["approach"], "deck")

    def test_inferred_height_field_alias_normalizes_to_auto(self) -> None:
        deck = parse_deck_text(
            """
            // Domain
            terr_voxel_height_field = Inferred
            """
        )

        config, sources = resolve_terrain_voxel_config(deck)

        self.assertEqual(config.height_field, "auto")
        self.assertEqual(sources["height_field"], "deck")

    def test_invalid_deck_values_fall_back_to_defaults(self) -> None:
        deck = parse_deck_text(
            """
            // Domain
            terr_voxel_approach = spline
            terr_voxel_grid_resolution = 0
            terr_voxel_idw_sigma = -1
            terr_voxel_idw_power = 0
            terr_voxel_idw_neighbors = 0
            """
        )

        warnings: list[str] = []
        config, sources = resolve_terrain_voxel_config(deck, warn=warnings.append)

        self.assertEqual(config.approach, "idw")
        self.assertEqual(config.grid_resolution, 50.0)
        self.assertEqual(config.idw_sigma, 1.0)
        self.assertEqual(config.idw_power, 2.0)
        self.assertEqual(config.idw_neighbors, 12)
        self.assertEqual(sources["approach"], "default")
        self.assertEqual(sources["grid_resolution"], "default")
        self.assertEqual(sources["idw_sigma"], "default")
        self.assertEqual(sources["idw_power"], "default")
        self.assertEqual(sources["idw_neighbors"], "default")
        self.assertGreaterEqual(len(warnings), 5)


if __name__ == "__main__":
    unittest.main()
