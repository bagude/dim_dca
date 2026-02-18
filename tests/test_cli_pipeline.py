from __future__ import annotations

import json
from pathlib import Path

from dim_dca.cli import main


def test_cli_pipeline(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("sys.argv", ["dim-dca", "--out", str(tmp_path)])
    main()
    assert (tmp_path / "synthetic.csv").exists()
    assert (tmp_path / "fit.png").exists()
    rows = json.loads((tmp_path / "model_comparison.json").read_text(encoding="utf-8"))
    assert len(rows) >= 3
