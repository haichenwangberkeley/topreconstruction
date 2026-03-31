import unittest

from triplet_ml import select_triplets


class TestSelectTripletsEfficiency(unittest.TestCase):
    def test_efficiency_with_reconstructible_and_non_reconstructible_events(self) -> None:
        summary = select_triplets._summarize_truth_efficiency_metrics(
            event_truth_matched_triplet_counts=[0, 2, 1],
            truth_matched_triplets_total=3,
            selected_truth_matched_triplets=2,
        )

        self.assertEqual(summary["events_total"], 3)
        self.assertEqual(summary["events_with_truth_matched_triplets"], 2)
        self.assertEqual(summary["events_without_truth_matched_triplets"], 1)
        self.assertAlmostEqual(summary["triplet_reconstruction_efficiency"], 2.0 / 3.0)

    def test_efficiency_when_no_truth_matched_triplets_exist(self) -> None:
        summary = select_triplets._summarize_truth_efficiency_metrics(
            event_truth_matched_triplet_counts=[0, 0, 0],
            truth_matched_triplets_total=0,
            selected_truth_matched_triplets=0,
        )

        self.assertEqual(summary["events_total"], 3)
        self.assertEqual(summary["events_with_truth_matched_triplets"], 0)
        self.assertEqual(summary["events_without_truth_matched_triplets"], 3)
        self.assertIsNone(summary["triplet_reconstruction_efficiency"])


if __name__ == "__main__":
    unittest.main()
