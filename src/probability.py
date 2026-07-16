def validate_probability(p):

    if not isinstance(p, (int, float)):

        raise TypeError(
            f"Probability must be numeric, got {type(p)}"
        )

    if p < 0.0 or p > 1.0:

        raise ValueError(
            f"Probability must be in [0, 1], got {p}"
        )

    return True


def conditional_probability(p_joint, p_evidence):

    validate_probability(p_joint)
    validate_probability(p_evidence)

    if p_evidence == 0.0:

        raise ValueError(
            "P(evidence) cannot be zero"
        )

    return p_joint / p_evidence


def joint_probability(*probabilities):

    result = 1.0

    for p in probabilities:

        validate_probability(p)

        result *= p

    return result


def marginal_probability(cpt, evidence, variable_states):

    total = 0.0

    for state in variable_states:

        key = (state, *evidence) if evidence else state

        p = cpt.get(key, 0.0)

        total += p

    return total


def normalize(probabilities):

    total = sum(probabilities.values())

    if total == 0.0:

        raise ValueError(
            "Cannot normalize: total probability is zero"
        )

    normalized = {}

    for key, value in probabilities.items():

        normalized[key] = value / total

    return normalized


def bayes_theorem(p_likelihood, p_prior, p_evidence):

    validate_probability(p_likelihood)
    validate_probability(p_prior)

    if p_evidence == 0.0:

        raise ValueError(
            "P(evidence) cannot be zero"
        )

    return (p_likelihood * p_prior) / p_evidence
