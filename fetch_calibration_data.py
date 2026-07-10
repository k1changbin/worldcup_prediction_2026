"""Build a compact, reproducible World Cup model-calibration dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx

from src.io_utils import atomic_write_json
from src.paths import data_path


TOURNAMENTS = {
    2018: {"host_code": "RU", "group_matches": 48},
    2022: {"host_code": "QA", "group_matches": 48},
}
RESULTS_URL = "https://www.eloratings.net/{year}_results.tsv"


def _parse_int(value):
    return int(str(value).replace("−", "-"))


def parse_world_cup_group_matches(tsv_text, year, host_code, group_matches=48):
    """Parse pre-match Elo and scores from an eloratings.net yearly TSV."""
    records = []
    for line in tsv_text.splitlines():
        parts = line.split("\t")
        if len(parts) < 12 or parts[7] != "WC":
            continue

        rating_change_a = _parse_int(parts[9])
        rating_after_a = _parse_int(parts[10])
        rating_after_b = _parse_int(parts[11])
        team_a = parts[3]
        team_b = parts[4]
        record = {
            "year": year,
            "date": f"{parts[0]}-{int(parts[1]):02d}-{int(parts[2]):02d}",
            "team_a": team_a,
            "team_b": team_b,
            "score_a": _parse_int(parts[5]),
            "score_b": _parse_int(parts[6]),
            # The yearly feed stores post-match ratings and team A's
            # zero-sum rating delta. Reversing that delta avoids leakage.
            "elo_a": rating_after_a - rating_change_a,
            "elo_b": rating_after_b + rating_change_a,
            "host_a": team_a == host_code,
            "host_b": team_b == host_code,
        }
        # eloratings.net normally puts the winner first, which would leak the
        # target into side assignment. Canonical code order is independent of
        # the result and makes W/L evaluation unbiased.
        if record["team_a"] > record["team_b"]:
            for left, right in (
                ("team_a", "team_b"),
                ("score_a", "score_b"),
                ("elo_a", "elo_b"),
                ("host_a", "host_b"),
            ):
                record[left], record[right] = record[right], record[left]
        records.append(record)
        if len(records) == group_matches:
            break

    if len(records) != group_matches:
        raise ValueError(
            f"Expected {group_matches} World Cup group matches for {year}, "
            f"found {len(records)}"
        )
    return records


def build_dataset(source_dir=None):
    records = []
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for year, config in TOURNAMENTS.items():
            if source_dir is None:
                response = client.get(
                    RESULTS_URL.format(year=year),
                    headers={"User-Agent": "worldcup-prediction-2026/0.2"},
                )
                response.raise_for_status()
                text = response.text
            else:
                source_path = Path(source_dir) / f"{year}_results.tsv"
                text = source_path.read_text(encoding="utf-8")
            records.extend(
                parse_world_cup_group_matches(
                    text,
                    year,
                    config["host_code"],
                    config["group_matches"],
                )
            )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        help="Read YEAR_results.tsv files locally instead of downloading them",
    )
    parser.add_argument(
        "--output",
        default=str(data_path("model_calibration_matches.json")),
    )
    args = parser.parse_args()

    records = build_dataset(args.source_dir)
    atomic_write_json(args.output, records)
    print(f"Saved {len(records)} leakage-free calibration matches to {args.output}")


if __name__ == "__main__":
    main()
