from src.bayesian_network import BayesianNetwork


def test_add_node():

    net = BayesianNetwork()

    net.add_node('Rain', ['Si', 'No'])

    assert 'Rain' in net.nodes

    assert len(net.nodes) == 1


def test_add_node_multiple_states():

    net = BayesianNetwork()

    net.add_node(
        'Weather', ['Sunny', 'Cloudy', 'Rain']
    )

    node = net.get_node('Weather')

    assert len(node.states) == 3


def test_add_edge():

    net = BayesianNetwork()

    net.add_node('Rain', ['Si', 'No'])
    net.add_node('Wet', ['Si', 'No'])

    net.add_edge('Rain', 'Wet')

    assert len(net.edges) == 1

    assert 'Rain' in net.get_parents('Wet')

    assert 'Wet' in net.get_children('Rain')


def test_set_cpt_prior():

    net = BayesianNetwork()

    net.add_node('Rain', ['Si', 'No'])

    cpt = {'Si': 0.2, 'No': 0.8}

    net.set_cpt('Rain', cpt)

    node = net.get_node('Rain')

    assert node.cpt == cpt


def test_set_cpt_conditional():

    net = BayesianNetwork()

    net.add_node('Rain', ['Si', 'No'])
    net.add_node('Wet', ['Si', 'No'])

    net.add_edge('Rain', 'Wet')

    cpt = {
        ('Si', 'Si'): 0.9,
        ('No', 'Si'): 0.1,
        ('Si', 'No'): 0.1,
        ('No', 'No'): 0.9,
    }

    net.set_cpt('Wet', cpt)

    node = net.get_node('Wet')

    assert node.cpt == cpt


def test_get_root_nodes():

    net = BayesianNetwork()

    net.add_node('A', ['X', 'Y'])
    net.add_node('B', ['X', 'Y'])
    net.add_node('C', ['X', 'Y'])

    net.add_edge('A', 'B')
    net.add_edge('A', 'C')

    roots = net.get_root_nodes()

    assert 'A' in roots

    assert 'B' not in roots

    assert 'C' not in roots


def test_is_valid_dag_true():

    net = BayesianNetwork()

    net.add_node('A', ['X'])
    net.add_node('B', ['X'])
    net.add_node('C', ['X'])

    net.add_edge('A', 'B')
    net.add_edge('B', 'C')

    assert net.is_valid_dag() is True


def test_is_valid_dag_cycle():

    net = BayesianNetwork()

    net.add_node('A', ['X'])
    net.add_node('B', ['X'])

    net.add_edge('A', 'B')
    net.add_edge('B', 'A')

    assert net.is_valid_dag() is False


def test_get_all_variables():

    net = BayesianNetwork()

    net.add_node('Rain', ['Si', 'No'])
    net.add_node('Sprinkler', ['Si', 'No'])
    net.add_node('Wet', ['Si', 'No'])

    variables = net.get_all_variables()

    assert len(variables) == 3

    assert 'Rain' in variables
    assert 'Sprinkler' in variables
    assert 'Wet' in variables
