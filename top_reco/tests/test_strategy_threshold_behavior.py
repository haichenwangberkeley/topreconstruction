import unittest

from triplet_ml import select_triplets


def _cand(i: int, score: float) -> select_triplets.TripletCandidate:
    return select_triplets.TripletCandidate(
        i=i,
        j=i + 1,
        k=i + 2,
        score=score,
        is_truth=False,
        triplet_pt=100.0,
        triplet_eta=0.1,
        triplet_phi=0.2,
        triplet_mass=172.0,
    )


class TestStrategyThresholdBehavior(unittest.TestCase):
    def setUp(self) -> None:
        self.triplets = [_cand(0, 0.9), _cand(3, 0.4), _cand(6, 0.1)]

    def test_top1_ignores_min_score_cut(self) -> None:
        selected = select_triplets._apply_strategy(
            triplets=self.triplets,
            strategy="top1",
            min_score=0.95,
            max_top_per_event=4,
            top_k=4,
            n_jets_in_event=9,
        )
        self.assertEqual(len(selected), 1)
        self.assertAlmostEqual(selected[0].score, 0.9)

    def test_topk_ignores_min_score_cut(self) -> None:
        selected = select_triplets._apply_strategy(
            triplets=self.triplets,
            strategy="topk",
            min_score=0.95,
            max_top_per_event=4,
            top_k=2,
            n_jets_in_event=9,
        )
        self.assertEqual(len(selected), 2)
        self.assertAlmostEqual(selected[0].score, 0.9)
        self.assertAlmostEqual(selected[1].score, 0.4)

    def test_threshold_strategy_respects_min_score(self) -> None:
        selected = select_triplets._apply_strategy(
            triplets=self.triplets,
            strategy="threshold",
            min_score=0.5,
            max_top_per_event=4,
            top_k=4,
            n_jets_in_event=9,
        )
        self.assertEqual(len(selected), 1)
        self.assertAlmostEqual(selected[0].score, 0.9)


if __name__ == "__main__":
    unittest.main()
