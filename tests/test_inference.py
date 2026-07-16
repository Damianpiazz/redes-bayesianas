from src.bayesian_network import BayesianNetwork
from src.inference import InferenceEngine


def build_simple_network():

    net = BayesianNetwork()

    net.add_node('Rain', ['Si', 'No'])
    net.add_node('Wet', ['Si', 'No'])

    net.add_edge('Rain', 'Wet')

    net.set_cpt('Rain', {'Si': 0.2, 'No': 0.8})

    net.set_cpt(
        'Wet',
        {
            ('Si', 'Si'): 0.9,
            ('No', 'Si'): 0.1,
            ('Si', 'No'): 0.3,
            ('No', 'No'): 0.7,
        },
    )

    return net


def test_query_no_evidence():

    net = build_simple_network()

    engine = InferenceEngine(net)

    result = engine.query('Rain')

    assert abs(result['Si'] - 0.2) < 1e-9
    assert abs(result['No'] - 0.8) < 1e-9


def test_query_with_evidence():

    net = build_simple_network()

    engine = InferenceEngine(net)

    result = engine.query(
        'Rain', {'Wet': 'Si'}
    )

    total = result['Si'] + result['No']

    assert abs(total - 1.0) < 1e-9

    assert abs(result['Si'] - 0.18 / 0.42) < 1e-9


def test_query_reverse():

    net = build_simple_network()

    engine = InferenceEngine(net)

    result = engine.query(
        'Wet', {'Rain': 'Si'}
    )

    total = result['Si'] + result['No']

    assert abs(total - 1.0) < 1e-9

    assert result['Si'] > result['No']


def test_joint_probability():

    net = build_simple_network()

    engine = InferenceEngine(net)

    assignment = {
        'Rain': 'Si',
        'Wet': 'Si',
    }

    result = engine.joint_probability(
        assignment
    )

    assert abs(result - 0.18) < 1e-9


def test_joint_probability_rain_no():

    net = build_simple_network()

    engine = InferenceEngine(net)

    assignment = {
        'Rain': 'No',
        'Wet': 'Si',
    }

    result = engine.joint_probability(
        assignment
    )

    assert abs(result - 0.24) < 1e-9


def test_mpe():

    net = build_simple_network()

    engine = InferenceEngine(net)

    assignment, prob = (
        engine.most_probable_explanation(
            {'Wet': 'Si'}
        )
    )

    assert assignment is not None

    assert prob > 0.0


def test_mpe_no_evidence():

    net = build_simple_network()

    engine = InferenceEngine(net)

    assignment, prob = (
        engine.most_probable_explanation()
    )

    assert 'Rain' in assignment
    assert 'Wet' in assignment

    assert prob > 0.0


def test_three_node_network():

    net = BayesianNetwork()

    net.add_node('Rain', ['Si', 'No'])
    net.add_node('Sprinkler', ['Si', 'No'])
    net.add_node('Wet', ['Si', 'No'])

    net.add_edge('Rain', 'Wet')
    net.add_edge('Sprinkler', 'Wet')

    net.set_cpt(
        'Rain', {'Si': 0.2, 'No': 0.8}
    )

    net.set_cpt(
        'Sprinkler', {'Si': 0.3, 'No': 0.7}
    )

    net.set_cpt(
        'Wet',
        {
            ('Si', 'Si', 'Si'): 0.95,
            ('No', 'Si', 'Si'): 0.05,
            ('Si', 'No', 'Si'): 0.8,
            ('No', 'No', 'Si'): 0.2,
            ('Si', 'Si', 'No'): 0.7,
            ('No', 'Si', 'No'): 0.3,
            ('Si', 'No', 'No'): 0.1,
            ('No', 'No', 'No'): 0.9,
        },
    )

    engine = InferenceEngine(net)

    result = engine.query(
        'Rain', {'Wet': 'Si'}
    )

    total = result['Si'] + result['No']

    assert abs(total - 1.0) < 1e-9
