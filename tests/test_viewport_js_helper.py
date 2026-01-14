from __future__ import annotations

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import importlib

cli_main = importlib.import_module("spectraqc.cli.main")


def _run_node(script_path: Path) -> None:
    result = subprocess.run(
        ["node", str(script_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"node script failed ({result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def test_viewport_math_zoom_pan() -> None:
    helper_source = cli_main.VIEWPORT_JS_HELPER
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        helper_path = tmp_path / "viewport_helper.js"
        helper_path.write_text(helper_source, encoding="utf-8")
        script_path = tmp_path / "viewport_test.js"
        script_path.write_text(
            """
const { ViewportMath } = require('./viewport_helper.js');
function assertClose(actual, expected, label) {
  if (Math.abs(actual - expected) > 1e-6) {
    throw new Error(`${label}: expected ${expected} got ${actual}`);
  }
}
const bounds = { xMin: 0, xMax: 10, yMin: 0, yMax: 20 };
const minSpan = { x: 1, y: 2 };
const vp = { xMin: 0, xMax: 10, yMin: 0, yMax: 20 };
const zoomed = ViewportMath.zoomViewport(vp, { x: 0.5, y: 0.5 }, { x: 5, y: 10 }, bounds, minSpan);
assertClose(zoomed.xMin, 2.5, 'zoom xMin');
assertClose(zoomed.xMax, 7.5, 'zoom xMax');
assertClose(zoomed.yMin, 5, 'zoom yMin');
assertClose(zoomed.yMax, 15, 'zoom yMax');
const panned = ViewportMath.panViewport(zoomed, { x: 2, y: -2 }, bounds);
assertClose(panned.xMin, 4.5, 'pan xMin');
assertClose(panned.xMax, 9.5, 'pan xMax');
assertClose(panned.yMin, 3, 'pan yMin');
assertClose(panned.yMax, 13, 'pan yMax');
const rect = ViewportMath.viewportFromRect({ x: -5, y: -5 }, { x: 3, y: 4 }, bounds, minSpan);
assertClose(rect.xMin, 0, 'rect xMin');
assertClose(rect.xMax, 8, 'rect xMax');
assertClose(rect.yMin, 0, 'rect yMin');
assertClose(rect.yMax, 9, 'rect yMax');
console.log('ok');
""",
            encoding="utf-8",
        )
        _run_node(script_path)
