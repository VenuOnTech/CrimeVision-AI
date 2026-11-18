from __future__ import annotations

import random

from pathlib import Path
from importlib.util import module_from_spec, spec_from_file_location

ADAPTER_PATH = Path(__file__).resolve().parents[1] / "datasets" / "adapt_uca.py"

spec = spec_from_file_location("crimevision_uca_adapter", ADAPTER_PATH)

if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load UCA adapter from {ADAPTER_PATH}")

adapter_module = module_from_spec(spec)
spec.loader.exec_module(adapter_module)

build_rows = adapter_module.build_rows


def test_uca_proxy_pairing_is_cross_category_and_kept_within_partition():
    records = [
        {"image_path": __import__("pathlib").Path("a.png"), "statement": "An assault occurs.", "class": "Assault", "stem": "Assault001"},
        {"image_path": __import__("pathlib").Path("b.png"), "statement": "A fire is visible.", "class": "Arson", "stem": "Arson001"},
        {"image_path": __import__("pathlib").Path("c.png"), "statement": "A theft occurs.", "class": "Stealing", "stem": "Stealing001"},
    ]
    generated = build_rows(records, random.Random(7))
    assert len(generated) == 6
    by_recording = {}
    for row in generated:
        by_recording.setdefault(row["source_recording_id"], []).append(row)
        if row["label"] == "MATCH":
            assert row["statement_source_recording_id"] == row["source_recording_id"]
            assert row["statement_source_environment_id"] == row["environment_id"]
        else:
            assert row["statement_source_recording_id"] != row["source_recording_id"]
            assert row["statement_source_environment_id"] != row["environment_id"]
            assert row["statement_source_recording_id"] in {r["stem"] for r in records}
    assert all({r["label"] for r in rows} == {"MATCH", "CONTRADICTION"} for rows in by_recording.values())
