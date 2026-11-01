"""Guard current author-approved diagram bytes; no model execution required."""
import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ApprovedArchitectureAssetTests(unittest.TestCase):
    def test_exact_master_and_current_detail_assets(self):
        manifest = json.loads((ROOT / 'docs/approved_assets.json').read_text())
        self.assertEqual(manifest['master_status'], 'AUTHOR_APPROVED_spacious_master_v3')
        self.assertEqual(manifest['current_chapter_iv_pages'], 50)
        self.assertEqual(len(manifest['assets']), 19)
        for asset in manifest['assets']:
            with self.subTest(asset=asset['repo_path']):
                path = ROOT / asset['repo_path']
                self.assertTrue(path.is_file())
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), asset['sha256'])

    def test_obsolete_renderer_and_figures_are_not_active(self):
        self.assertFalse((ROOT / 'scripts/render_architecture.py').exists())
        self.assertFalse((ROOT / 'scripts/run_development_tuning_v2.sh').exists())
        self.assertFalse((ROOT / 'scripts/summarize_development_tuning_v2.py').exists())
        self.assertFalse(list((ROOT / 'docs/figures').glob('*_v2.*')))
        architecture = (ROOT / 'docs/SYSTEM_ARCHITECTURE.md').read_text()
        self.assertNotIn('```mermaid', architecture)
        self.assertIn('figures/approved/architecture_master.pdf', architecture)
        self.assertIn('spacious master V3', architecture)


if __name__ == '__main__':
    unittest.main()
