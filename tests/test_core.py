from careerbench.core import dataset, evaluate


def test_shapes():
    x, y, group = dataset(100)
    assert x.shape == (100, 24)
    assert y.shape == (100, 6)
    assert len(group) == 100


def test_metrics_are_bounded():
    for record in evaluate(300):
        assert 0 <= record["micro_f1"] <= 1
        assert 0 <= record["hamming_loss"] <= 1
