import csv
from pathlib import Path
import tempfile
import unittest

from evaluate_ppo_nominal import METRICS, paper_surface_fluctuation, read_metrics, sha256


class NominalResultTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "metrics.csv"
        self.row = {"severity": "off", **{metric: "0.1" for metric in METRICS}}

    def write_rows(self, rows: list[dict[str, str]]) -> None:
        with self.path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.row))
            writer.writeheader()
            writer.writerows(rows)

    def test_reads_nominal_metrics(self) -> None:
        self.write_rows([self.row])
        self.assertEqual(read_metrics(self.path), {metric: 0.1 for metric in METRICS})

    def test_rejects_missing_or_multiple_conditions(self) -> None:
        for rows in ([], [self.row, self.row], [{**self.row, "severity": "light"}]):
            with self.subTest(rows=rows):
                self.write_rows(rows)
                with self.assertRaises(ValueError):
                    read_metrics(self.path)

    def test_rejects_invalid_values(self) -> None:
        for value in ("nan", "inf", "-0.1", "not-a-number", ""):
            with self.subTest(value=value):
                self.write_rows([{**self.row, "avg_rmse": value}])
                with self.assertRaises(ValueError):
                    read_metrics(self.path)

    def test_rejects_changed_schema(self) -> None:
        self.path.write_text("severity,avg_rmse\noff,0.1\n")
        with self.assertRaises(ValueError):
            read_metrics(self.path)

    def test_hashes_file_bytes(self) -> None:
        self.path.write_bytes(b"abc")
        self.assertEqual(
            sha256(self.path),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )

    def test_paper_fluctuation_uses_only_two_surfaces(self) -> None:
        metrics = {"roll_fcs_fluct": 0.01, "pitch_fcs_fluct": 0.03, "avg_fcs_fluct": 99.0}
        self.assertAlmostEqual(paper_surface_fluctuation(metrics), 0.02)


if __name__ == "__main__":
    unittest.main()
