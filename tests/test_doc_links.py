"""Every local link and image in the docs, README and notebooks must point to a file in the repository."""
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\]\(([^)\s#]+)")
# Folders kept out of Git; links into them are allowed to be missing.
NOT_IN_GIT = ("paper/", "output/", "results/", "data/", "data_5_fold/")


def markdown_sources():
    for path in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
        yield path, path.read_text(encoding="utf-8")
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        yield path, "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "markdown")


class DocLinkTests(unittest.TestCase):
    def test_local_links_resolve(self):
        for path, text in markdown_sources():
            for target in LINK.findall(text):
                if "://" in target or target.startswith("mailto:"):
                    continue
                resolved = (path.parent / target).resolve()
                if resolved.relative_to(ROOT).as_posix().startswith(NOT_IN_GIT):
                    continue
                with self.subTest(file=path.relative_to(ROOT).as_posix(), link=target):
                    self.assertTrue(resolved.exists())

    def test_every_figure_is_shown_with_its_svg(self):
        architecture = (ROOT / "docs/SYSTEM_ARCHITECTURE.md").read_text(encoding="utf-8")
        for figure in (ROOT / "docs/figures").glob("*.png"):
            with self.subTest(figure=figure.name):
                self.assertIn(f"figures/{figure.name}", architecture)
                self.assertTrue(figure.with_suffix(".svg").is_file())


if __name__ == "__main__":
    unittest.main()
