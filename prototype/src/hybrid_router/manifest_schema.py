from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List


FIELDNAMES = [
    "sample_id",
    "split",
    "label",
    "protection",
    "degradation_type",
    "degradation_level",
    "source_path",
    "planned_output_path",
    "identity_id",
    "provenance_key",
    "passive_key",
    "pair_group_id",
    "notes",
]


@dataclass
class SampleRecord:
    sample_id: str
    split: str
    label: str
    protection: str
    degradation_type: str
    degradation_level: str
    source_path: str
    planned_output_path: str
    identity_id: str
    provenance_key: str
    passive_key: str
    pair_group_id: str
    notes: str = ""


def write_manifest(path: Path, records: Iterable[SampleRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def load_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    missing = [field for field in FIELDNAMES if field not in reader.fieldnames]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    return rows
