import csv
from pathlib import Path
import subprocess
import tempfile
import unittest

from evaluate_ppo_nominal import (
    METRICS,
    REFERENCE_SETS,
    evaluation_log_problems,
    paper_surface_fluctuation,
    read_metrics,
    runtime_package_info,
    sha256,
    simulator_source_files,
)


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


class SimulatorProfileTests(unittest.TestCase):
    @staticmethod
    def log(channels: int, profile_line: str = "") -> str:
        return (
            "Episode reward: 1\n" * 25 + "e_obs shape: (50000, 14)\n"
            + f"eps_fcs_fluct shape: (25, {channels})\n" + profile_line
        )

    def test_accepts_expected_logs(self) -> None:
        self.assertEqual(evaluation_log_problems(self.log(3), "current"), [])
        self.assertEqual(
            evaluation_log_problems(self.log(2, "Simulator profile: icinco2024\n"), "icinco2024"),
            [],
        )

    def test_rejects_wrong_channel_count(self) -> None:
        self.assertTrue(evaluation_log_problems(self.log(2), "current"))
        self.assertTrue(
            evaluation_log_problems(self.log(3, "Simulator profile: icinco2024\n"), "icinco2024")
        )

    def test_rejects_unconfirmed_historical_profile(self) -> None:
        self.assertTrue(evaluation_log_problems(self.log(2), "icinco2024"))

    def test_rejects_incomplete_or_unbounded_runs(self) -> None:
        for log_text in (
            self.log(3).replace("Episode reward: 1\n", "", 1),
            self.log(3) + "Out of bounds\n",
            self.log(3).replace("(50000, 14)", "(49999, 14)"),
        ):
            with self.subTest(log_text=log_text[-40:]):
                self.assertTrue(evaluation_log_problems(log_text, "current"))


class ReferenceSetTests(unittest.TestCase):
    def test_reference_set_files_match_historical_names(self) -> None:
        self.assertEqual(
            REFERENCE_SETS,
            {
                "easy": ("easy", "easyrefs"),
                "medium": ("medium", "medrefs"),
                "hard": ("hard", "hardrefs"),
            },
        )


class RuntimeProvenanceTests(unittest.TestCase):
    def test_records_imported_version_separately_from_distribution_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample_module.py").write_text('__version__ = "2.0"\n')
            metadata = root / "sample_distribution-1.0.dist-info"
            metadata.mkdir()
            (metadata / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: sample-distribution\nVersion: 1.0\n"
            )
            result = runtime_package_info(
                {"PYTHONPATH": str(root)}, {"sample-distribution": "sample_module"},
            )
            self.assertEqual(result["packages"]["sample-distribution"], "2.0")
            self.assertEqual(result["distribution_metadata"]["sample-distribution"], "1.0")
            self.assertEqual(result["distribution_metadata_missing"], [])
            self.assertEqual(
                result["runtime_module_paths"]["sample-distribution"], str(root / "sample_module.py"),
            )

    def test_records_missing_distribution_metadata_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample_module.py").write_text('__version__ = "2.0"\n')
            result = runtime_package_info(
                {"PYTHONPATH": str(root)}, {"sample-distribution": "sample_module"},
            )
            self.assertEqual(result["packages"]["sample-distribution"], "2.0")
            self.assertIsNone(result["distribution_metadata"]["sample-distribution"])
            self.assertEqual(result["distribution_metadata_missing"], ["sample-distribution"])

    def test_probes_with_the_evaluator_directory_first_on_the_import_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample_module.py").write_text('__version__ = "1.0"\n')
            entrypoint = root / "evaluator" / "evaluate.py"
            entrypoint.parent.mkdir()
            (entrypoint.parent / "sample_module.py").write_text('__version__ = "2.0"\n')
            result = runtime_package_info(
                {"PYTHONPATH": str(root)}, {"sample-distribution": "sample_module"},
                entrypoint=entrypoint,
            )
            self.assertEqual(result["packages"]["sample-distribution"], "2.0")


class SimulatorSourceFileTests(unittest.TestCase):
    def test_includes_untracked_but_not_ignored_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            package = root / "fw_jsbgym"
            package.mkdir()
            (root / ".gitignore").write_text("*.log\n")
            for name in ("tracked.py", "untracked.py", "ignored.log"):
                (package / name).write_text(name)
            subprocess.run(["git", "-C", str(root), "add", "fw_jsbgym/tracked.py"], check=True)
            self.assertEqual(
                simulator_source_files(root),
                [package / "tracked.py", package / "untracked.py"],
            )


if __name__ == "__main__":
    unittest.main()
