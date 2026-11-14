"""The neural-network filter (vs_mlfilter): fingerprints, the area under the curve, training on a made-up problem that has
a real answer (molecules with a phenyl ring dock better), saving, and the safeguards. Nothing here needs the cluster."""
import numpy as np
import pytest

import vs_mlfilter


def synthetic(n, seed=0, noise=0.3):
    """Random chains of C, N and O; 30% carry a phenyl ring at the end. Those score about -9, the others about -6.
    Returns (SMILES, scores, has_phenyl)."""
    rng = np.random.default_rng(seed)
    smiles, scores, phenyl = [], [], []
    for _ in range(n):
        chain = "".join(rng.choice(list("CCCCNO"), size=int(rng.integers(4, 11))))
        has = bool(rng.random() < 0.3)
        smiles.append(chain + ("c1ccccc1" if has else ""))
        scores.append((-9.0 if has else -6.0) + float(rng.normal(0, noise)))
        phenyl.append(has)
    return smiles, np.array(scores), np.array(phenyl)


@pytest.fixture(scope="module")
def trained():
    smiles, scores, _ = synthetic(700, seed=1)
    return vs_mlfilter.train(smiles, scores, cutoff=0.5, seed=3)


class TestAuc:
    def test_a_perfect_ranking(self):
        assert vs_mlfilter.auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0

    def test_a_reversed_ranking(self):
        assert vs_mlfilter.auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0

    def test_ties_count_half(self):
        assert vs_mlfilter.auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5

    def test_a_known_value(self):
        # positives 0.8 and 0.4, negatives 0.6 and 0.2: three of the four pairs are in order
        assert vs_mlfilter.auc([1, 1, 0, 0], [0.8, 0.4, 0.6, 0.2]) == 0.75

    def test_one_class_has_no_auc(self):
        assert vs_mlfilter.auc([1, 1], [0.2, 0.3]) is None


class TestFingerprints:
    def test_shape_and_values(self):
        x, ok = vs_mlfilter.fingerprints(["CCO", "c1ccccc1"])
        assert x.shape == (2, 1024) and x.dtype == np.float32 and ok.all()
        assert set(np.unique(x)) <= {0.0, 1.0} and x[0].sum() > 0 and not np.array_equal(x[0], x[1])

    def test_unreadable_and_empty_smiles_are_marked(self):
        x, ok = vs_mlfilter.fingerprints(["CCO", "not a smiles", ""])
        assert list(ok) == [True, False, False] and x[1:].sum() == 0

    def test_the_same_molecule_gives_the_same_fingerprint(self):
        a, _ = vs_mlfilter.fingerprints(["OCC"])
        b, _ = vs_mlfilter.fingerprints(["CCO"])
        assert np.array_equal(a, b)


class TestTraining:
    def test_it_learns_a_real_pattern_and_the_report_says_so(self, trained):
        report = trained.report
        assert report["auc"] > 0.95
        assert report["recall"] > 0.8 and report["kept"] < 0.6
        assert report["forms"] == 700 and report["positives"] >= 160
        assert report["held_back"] >= 100 and report["trained_on"] > 100

    def test_the_positives_are_the_best_quarter(self, trained):
        smiles, scores, _ = synthetic(700, seed=1)
        assert trained.threshold == pytest.approx(np.percentile(scores, 25))

    def test_unseen_molecules_with_the_pattern_score_higher(self, trained):
        smiles, _scores, phenyl = synthetic(300, seed=99)
        p = trained.predict(smiles)
        assert p[phenyl].mean() > 0.8 and p[~phenyl].mean() < 0.3
        assert vs_mlfilter.auc(phenyl, p) > 0.95

    def test_the_same_seed_gives_the_same_model(self):
        smiles, scores, _ = synthetic(250, seed=4)
        a = vs_mlfilter.train(smiles, scores, seed=5)
        b = vs_mlfilter.train(smiles, scores, seed=5)
        assert np.array_equal(a.predict(smiles), b.predict(smiles)) and a.report["auc"] == b.report["auc"]

    def test_scores_that_mean_nothing_give_a_model_near_chance(self):
        smiles, _scores, _ = synthetic(600, seed=6)
        noise = np.random.default_rng(7).normal(-7, 1, len(smiles))
        assert vs_mlfilter.train(smiles, noise, seed=1).report["auc"] < 0.75

    def test_the_training_set_is_capped(self):
        smiles, scores, _ = synthetic(400, seed=8)
        assert vs_mlfilter.train(smiles, scores, max_train=120, seed=1).report["forms"] == 120

    def test_unreadable_smiles_are_left_out_of_the_training(self):
        smiles, scores, _ = synthetic(300, seed=9)
        smiles[0], smiles[1] = "garbage", ""
        assert vs_mlfilter.train(smiles, scores, seed=1).report["forms"] == 298

    def test_too_few_molecules_are_refused(self):
        with pytest.raises(ValueError, match="fewer than 20"):
            vs_mlfilter.train(["CCO"] * 10, np.arange(10.0))

    def test_a_single_class_is_refused(self):
        with pytest.raises(ValueError):
            vs_mlfilter.train(["CCO"] * 40, np.zeros(40) - 5.0)


class TestModelFiles:
    def test_a_saved_model_predicts_the_same(self, trained, tmp_path):
        smiles, _s, _p = synthetic(50, seed=11)
        path = str(tmp_path / "model.npz")
        trained.save(path)
        loaded = vs_mlfilter.Model.load(path)
        assert np.array_equal(trained.predict(smiles), loaded.predict(smiles))
        assert loaded.threshold == pytest.approx(trained.threshold)

    def test_unreadable_smiles_get_no_probability(self, trained):
        p = trained.predict(["CCO", "garbage", "", "c1ccccc1"])
        assert np.isnan(p[1]) and np.isnan(p[2]) and not np.isnan(p[0]) and not np.isnan(p[3])

    def test_probabilities_are_between_0_and_1(self, trained):
        smiles, _s, _p = synthetic(100, seed=12)
        p = trained.predict(smiles)
        assert p.min() >= 0 and p.max() <= 1

    def test_the_prediction_does_not_depend_on_the_batch_size(self, trained, monkeypatch):
        smiles, _s, _p = synthetic(60, seed=13)
        whole = trained.predict(smiles)
        monkeypatch.setattr(vs_mlfilter, "PREDICT_BATCH", 7)
        assert np.allclose(trained.predict(smiles), whole, atol=1e-6)
