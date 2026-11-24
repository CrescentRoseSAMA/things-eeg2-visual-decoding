"""Packaging must refuse an incomplete run or an unsafe archive name."""
import tempfile
import unittest
from pathlib import Path

from package_experiment import archive_label, check_summary, create_archive, report_files, paired_rows


class PackageTests(unittest.TestCase):
    def test_summary_permits_roundoff_but_rejects_changed_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overall.csv"
            path.write_text("config,subject,sd\nab,1,0.5867282164682381\n", encoding="utf-8")
            expected = dict(config="ab", subject=1, sd=0.5867282164682383)
            check_summary(path, [expected])
            for change in (dict(config="a"), dict(subject=2), dict(sd=0.58673)):
                with self.assertRaisesRegex(ValueError, "Stale result"):
                    check_summary(path, [{**expected, **change}])
    def test_evidence_can_omit_word_while_final_requires_it(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluation = Path(directory)
            self.assertEqual(report_files(evaluation, evidence_only=True), set())
            with self.assertRaisesRegex(ValueError, "Word"):
                report_files(evaluation, evidence_only=False)

    def test_incomplete_followup_table_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            table = Path(directory) / "paired_runs.csv"
            table.write_text("subject,seed,top1_pct\n1,0,10\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "50"):
                paired_rows(table)
    def test_missing_runs_create_no_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "0/200"):
                create_archive(root, root / "dist", "学号+姓名")
            self.assertFalse((root / "dist").exists())

    def test_invalid_student_name_is_rejected(self):
        with self.assertRaises(ValueError):
            archive_label("001", "../姓名")
        self.assertEqual(archive_label("", ""), "待填写学号+待填写姓名")
        self.assertEqual(archive_label("001", "张三"), "001+张三")


if __name__ == "__main__":
    unittest.main()
