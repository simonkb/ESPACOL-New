from __future__ import annotations

import gzip

from tools.build_origin_acceptance_manifest import (
    _anonymous_id,
    _membership_rows,
    _write_deterministic_csv_gz,
)


def test_membership_export_is_anonymous_and_deterministic(tmp_path) -> None:
    items = [
        ("/private/data/100_left.jpeg", 1),
        ("/private/data/100_right.jpeg", 2),
    ]
    rows = _membership_rows("dr", (("outer_test", items),))
    assert len(rows) == 2
    assert all("private" not in str(row) for row in rows)
    assert rows[0]["patient_cluster_sha256"] == rows[1]["patient_cluster_sha256"]
    assert rows[0]["patient_cluster_sha256"] == _anonymous_id("dr", "100")

    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"
    _write_deterministic_csv_gz(first, rows)
    _write_deterministic_csv_gz(second, rows)
    assert first.read_bytes() == second.read_bytes()
    with gzip.open(first, "rt", encoding="utf-8") as stream:
        content = stream.read()
    assert "image_id_sha256" in content
    assert "100_left" not in content

