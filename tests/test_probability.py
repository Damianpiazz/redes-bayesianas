import pytest

from src.probability import (
    validate_probability,
    conditional_probability,
    joint_probability,
    normalize,
    bayes_theorem,
)


def test_validate_probability_valid():

    assert validate_probability(0.0) is True
    assert validate_probability(0.5) is True
    assert validate_probability(1.0) is True


def test_validate_probability_invalid():

    with pytest.raises(ValueError):

        validate_probability(-0.1)

    with pytest.raises(ValueError):

        validate_probability(1.1)


def test_validate_probability_type_error():

    with pytest.raises(TypeError):

        validate_probability("abc")


def test_conditional_probability_basic():

    result = conditional_probability(
        0.2, 0.8
    )

    assert abs(result - 0.25) < 1e-9


def test_conditional_probability_zero_evidence():

    with pytest.raises(ValueError):

        conditional_probability(0.5, 0.0)


def test_joint_probability_basic():

    result = joint_probability(0.5, 0.5, 0.5)

    assert abs(result - 0.125) < 1e-9


def test_joint_probability_single():

    result = joint_probability(0.3)

    assert abs(result - 0.3) < 1e-9


def test_joint_probability_invalid():

    with pytest.raises(ValueError):

        joint_probability(1.5)


def test_normalize_basic():

    probs = {'A': 0.3, 'B': 0.7}

    result = normalize(probs)

    assert abs(result['A'] - 0.3) < 1e-9
    assert abs(result['B'] - 0.7) < 1e-9

    total = sum(result.values())

    assert abs(total - 1.0) < 1e-9


def test_normalize_unequal():

    probs = {'A': 0.1, 'B': 0.3, 'C': 0.6}

    result = normalize(probs)

    total = sum(result.values())

    assert abs(total - 1.0) < 1e-9

    assert result['C'] > result['B']
    assert result['B'] > result['A']


def test_normalize_zero_total():

    with pytest.raises(ValueError):

        normalize({'A': 0.0, 'B': 0.0})


def test_bayes_theorem_basic():

    result = bayes_theorem(
        p_likelihood=0.8,
        p_prior=0.1,
        p_evidence=0.2,
    )

    assert abs(result - 0.4) < 1e-9


def test_bayes_theorem_zero_evidence():

    with pytest.raises(ValueError):

        bayes_theorem(0.5, 0.5, 0.0)
