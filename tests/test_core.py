from careerbench.core import (
    DOMAIN_LABELS,
    POSITION_LABELS,
    binary_entropy,
    evaluate,
    generate_reference_dataset,
    privacy_audit,
    select_features,
)


def test_paper_dimensions():
    data = generate_reference_dataset()
    assert data.features.shape == (420, 26)
    assert data.targets.shape == (420, 2)
    assert len(DOMAIN_LABELS) == 6
    assert len(POSITION_LABELS) == 8
    assert len(select_features(data.features, data.targets)) == 11


def test_entropy_bounds():
    data = generate_reference_dataset()
    assert all(0 <= binary_entropy(data.features[name].to_numpy()) <= 1 for name in data.features)


def test_all_six_classifiers_and_metrics():
    records, predictions, selected = evaluate()
    assert len(records) == 6
    assert len(predictions) == 120
    assert len(selected) == 11
    for record in records:
        assert 0 <= record["domain_accuracy"] <= 1
        assert 0 <= record["position_accuracy"] <= 1
        assert 0 <= record["joint_accuracy"] <= 1


def test_privacy_audit_blocks_identifier_and_fairness_overclaim():
    audit = privacy_audit(generate_reference_dataset())
    assert audit["synthetic_fixture"]
    assert not audit["direct_identifier_columns_present"]
    assert not audit["sensitive_attribute_columns_present"]
    assert not audit["private_records_redistributed"]
    assert not audit["fairness_metrics_claimed"]
