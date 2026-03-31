import unittest

from triplet_ml import strategy_compare


class TestStrategyComparison(unittest.TestCase):
    def test_build_rows_and_rank_by_efficiency(self) -> None:
        reports = {
            "greedy_disjoint": {
                "triplet_reconstruction_efficiency": 0.20,
                "selected_truth_matched_triplets": 10,
                "truth_matched_triplets_total": 50,
                "selected_rows_total": 40,
            },
            "top1": {
                "triplet_reconstruction_efficiency": 0.18,
                "selected_truth_matched_triplets": 9,
                "truth_matched_triplets_total": 50,
                "selected_rows_total": 25,
            },
            "threshold": {
                "triplet_reconstruction_efficiency": None,
                "selected_truth_matched_triplets": 0,
                "truth_matched_triplets_total": 0,
                "selected_rows_total": 0,
            },
        }

        rows = strategy_compare.build_strategy_comparison_rows(reports)

        self.assertEqual(rows[0]["strategy"], "greedy_disjoint")
        self.assertEqual(rows[0]["rank_by_efficiency"], 1)
        self.assertEqual(rows[1]["strategy"], "top1")
        self.assertEqual(rows[1]["rank_by_efficiency"], 2)
        self.assertEqual(rows[2]["strategy"], "threshold")
        self.assertIsNone(rows[2]["triplet_reconstruction_efficiency"])
        self.assertIsNone(rows[2]["rank_by_efficiency"])

    def test_tie_breaker_uses_selected_truth_count(self) -> None:
        reports = {
            "A": {
                "triplet_reconstruction_efficiency": 0.20,
                "selected_truth_matched_triplets": 10,
                "truth_matched_triplets_total": 50,
                "selected_rows_total": 30,
            },
            "B": {
                "triplet_reconstruction_efficiency": 0.20,
                "selected_truth_matched_triplets": 11,
                "truth_matched_triplets_total": 55,
                "selected_rows_total": 31,
            },
        }

        rows = strategy_compare.build_strategy_comparison_rows(reports)
        self.assertEqual(rows[0]["strategy"], "B")
        self.assertEqual(rows[0]["rank_by_efficiency"], 1)
        self.assertEqual(rows[1]["strategy"], "A")
        self.assertEqual(rows[1]["rank_by_efficiency"], 2)


if __name__ == "__main__":
    unittest.main()
