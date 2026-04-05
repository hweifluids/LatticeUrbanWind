from __future__ import annotations

from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = REPO_ROOT / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from deck_io import parse_deck_text


class DeckIoTests(unittest.TestCase):
    def test_bool_tokens_are_fuzzy_and_quoted(self) -> None:
        deck = parse_deck_text(
            """
            // Physics
            buoyancy = "yes"
            coriolis_term = t
            ibm_enabler = n
            enable_top_sponge = 0
            enable_buffer_nudging = 2
            """
        )

        self.assertTrue(deck.get_bool("buoyancy"))
        self.assertTrue(deck.get_bool("coriolis_term"))
        self.assertFalse(deck.get_bool("ibm_enabler"))
        self.assertFalse(deck.get_bool("enable_top_sponge"))
        self.assertTrue(deck.get_bool("enable_buffer_nudging"))

    def test_alias_keys_normalize_to_canonical_keys(self) -> None:
        deck = parse_deck_text(
            """
            // Turbulence inflow
            vk-inlet-enable = "y"
            vk inlet anisotropy scale = [1.0, 2.0, 3.0]
            """
        )

        self.assertTrue(deck.has("turb_inflow_enable"))
        self.assertTrue(deck.get_bool("turb_inflow_enable"))
        self.assertEqual(deck.get_float_list("vk_inlet_anisotropy"), [1.0, 2.0, 3.0])

    def test_render_repairs_order_sections_and_preserves_unknowns(self) -> None:
        deck = parse_deck_text(
            """
            custom_note = alpha
            probes =
            // CFD control
            gpu_memory = 24000
            vk_inlet_enable = yes
            mystery-key = 42
            // Domain
            cut_lon_manual = [121.7, 121.3]
            cut_lat_manual = [31.4, 31.1]
            """
        )
        deck.set_bool("flux_correction", True)
        rendered = deck.render()

        self.assertIn("// Domain", rendered)
        self.assertIn("// CFD Controls", rendered)
        self.assertIn("// Output & Probes", rendered)
        self.assertIn("probes =", rendered)
        self.assertIn("turb_inflow_enable = true", rendered)
        self.assertIn("mystery_key = 42", rendered)
        self.assertLess(rendered.index("// Domain"), rendered.index("// CFD Controls"))
        self.assertLess(rendered.index("// CFD Controls"), rendered.index("// Output & Probes"))

    def test_terrain_voxel_keys_are_known_and_round_trip(self) -> None:
        deck = parse_deck_text(
            """
            // Domain
            terr_voxel_height_field = HEIGHT_M
            terr_voxel_ignore_under = 3.500000
            terr_voxel_approach = kriging_gpu
            terr_voxel_grid_resolution = 25.000000
            terr_voxel_idw_sigma = 0.500000
            terr_voxel_idw_power = 1.500000
            terr_voxel_idw_neighbors = 8
            """
        )

        self.assertEqual(deck.get_text("terr_voxel_height_field"), "HEIGHT_M")
        self.assertEqual(deck.get_float("terr_voxel_ignore_under"), 3.5)
        self.assertEqual(deck.get_text("terr_voxel_approach"), "kriging_gpu")
        self.assertEqual(deck.get_float("terr_voxel_grid_resolution"), 25.0)
        self.assertEqual(deck.get_float("terr_voxel_idw_sigma"), 0.5)
        self.assertEqual(deck.get_float("terr_voxel_idw_power"), 1.5)
        self.assertEqual(deck.get_int("terr_voxel_idw_neighbors"), 8)

        rendered = deck.render()
        self.assertIn("terr_voxel_approach = kriging_gpu", rendered)
        self.assertIn("terr_voxel_idw_neighbors = 8", rendered)


if __name__ == "__main__":
    unittest.main()
