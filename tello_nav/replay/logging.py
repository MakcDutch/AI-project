import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def plain(value: Any) -> Any:
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def write_results(path: str | Path, records: list[dict], metadata: dict) -> None:
    """CSV nested objects are JSON cells; metadata always has a sidecar JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps({"metadata": plain(metadata), "records": records},
                                   indent=2, allow_nan=False) + "\n")
    elif path.suffix == ".csv":
        with path.open("w", newline="") as handle:
            if records:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
                for record in records:
                    writer.writerow({key: json.dumps(value, allow_nan=False)
                                     if isinstance(value, (dict, list)) else value
                                     for key, value in record.items()})
        path.with_suffix(".metadata.json").write_text(
            json.dumps(plain(metadata), indent=2, allow_nan=False) + "\n")
    else:
        raise ValueError("log path must end with .json or .csv")
