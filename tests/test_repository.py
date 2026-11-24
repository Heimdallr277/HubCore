"""Repository consistency: version, dependencies, package boundaries and documentation."""

import re
import subprocess
import sys
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

from packaging.requirements import Requirement

import hubcore
from hubcore import robustness as R
from hubcore.park import PRESETS

ROOT = Path(__file__).resolve().parents[1]


def test_one_version_everywhere():
    version = hubcore.__version__
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "hubcore.__version__"}
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert re.search(rf"^version: {re.escape(version)}$", citation, re.M)


def test_pinned_requirements_satisfy_the_package_ranges():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    ranges = {Requirement(r).name: Requirement(r).specifier
              for r in pyproject["dependencies"] + pyproject["optional-dependencies"]["app"]}
    pins = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            req = Requirement(line)
            pins[req.name] = next(iter(req.specifier)).version
    assert set(pins) == set(ranges)
    for name, version in pins.items():
        assert ranges[name].contains(version), (name, version, str(ranges[name]))


def test_science_package_imports_without_interface_libraries():
    """src/hubcore needs only NumPy, SciPy and pandas; Streamlit and Plotly stay in ui/."""
    code = (
        "import sys, importlib, pkgutil\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('streamlit', 'plotly'):\n"
        "            raise ImportError('blocked ' + name)\n"
        "sys.meta_path.insert(0, Block())\n"
        f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
        "import hubcore\n"
        "for m in pkgutil.iter_modules(hubcore.__path__):\n"
        "    importlib.import_module('hubcore.' + m.name)\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr


def test_bundled_ensembles_use_the_shared_file_names():
    for name in PRESETS:
        assert (ROOT / "data/precomputed" / R.preset_filename(name, 200, 42)).exists(), name


def test_documentation_links_resolve():
    for document in [ROOT / "README.md", ROOT / "data/PROVENANCE.md", *(ROOT / "docs").glob("*.md")]:
        links = re.findall(r"\[[^\]\n]*\]\(([^)\s]+)\)", document.read_text(encoding="utf-8"))
        for link in links:
            url = urlsplit(link)
            if not url.scheme and url.path:
                assert (document.parent / unquote(url.path)).is_file(), (document.name, link)
