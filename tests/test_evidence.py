import numpy as np

from reader_provenance.evaluation.evidence import accumulate_log_posteriors


def test_single_observation_is_unchanged() -> None:
    logp = np.log(np.asarray([[0.2, 0.8]]))
    actual = accumulate_log_posteriors(logp)
    assert np.allclose(np.exp(actual), [[0.2, 0.8]])


def test_nonuniform_prior_is_removed_for_repeated_evidence() -> None:
    prior = np.log(np.asarray([0.8, 0.2]))
    observations = np.stack([prior, prior])
    actual = accumulate_log_posteriors(observations, log_prior=prior)
    assert np.allclose(np.exp(actual), [0.8, 0.2])


def test_independent_evidence_accumulates() -> None:
    observations = np.log(np.asarray([[0.6, 0.4], [0.7, 0.3]]))
    actual = np.exp(accumulate_log_posteriors(observations))
    expected = np.asarray([0.6 * 0.7, 0.4 * 0.3])
    expected /= expected.sum()
    assert np.allclose(actual, expected)
