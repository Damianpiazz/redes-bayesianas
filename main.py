import sys

from rich.console import Console
from rich.table import Table

from src.bayesian_network import BayesianNetwork
from src.inference import InferenceEngine
from src.utils import (
    load_csv,
    build_prior,
    build_conditional_multi,
    format_probability,
)

from src.visualization import (
    plot_network_graph,
    plot_cpt_heatmap,
    plot_cpt_all,
    plot_probability_comparison,
    print_network_ascii,
)


console = Console()


DATASETS = {
    'weather': {
        'name': 'Weather (Rain, Sprinkler, Wet Grass)',
        'path': 'data/weather.csv',
        'query_var': 'Lluvia',
        'evidence': {'Cesped_Mojado': 'Si'},
        'structure': [
            ('Lluvia', 'Aspersor'),
            ('Lluvia', 'Cesped_Mojado'),
            ('Aspersor', 'Cesped_Mojado'),
        ],
    },
}


def header(text):

    console.print(f"\n[bold]{text}[/bold]")
    console.print("-" * len(text))


def print_cpt_table(name, cpt, is_prior):

    table = Table(
        show_header=True,
    )

    if is_prior:

        table.add_column(name.upper())
        table.add_column(
            "PROBABILIDAD", justify="right"
        )

        for state, prob in cpt.items():

            table.add_row(
                state,
                format_probability(prob),
            )

    else:

        sample_key = next(iter(cpt))
        n_parents = len(sample_key) - 1

        for i in range(n_parents):

            table.add_column(
                f"PADRE_{i+1}"
            )

        table.add_column(name.upper())
        table.add_column(
            "P(CONDICIONAL)", justify="right"
        )

        for key, prob in cpt.items():

            child_state = key[0]
            parent_states = key[1:]

            row = list(parent_states) + [
                child_state,
                format_probability(prob),
            ]

            table.add_row(*row)

    console.print(table)


def print_query_result(query_var, result):

    table = Table(
        show_header=True,
    )

    table.add_column(f"P({query_var})")
    table.add_column(
        "PROBABILIDAD", justify="right"
    )

    for state, prob in result.items():

        table.add_row(
            state,
            format_probability(prob),
        )

    console.print(table)


def build_network(config):

    data = load_csv(config['path'])

    network = BayesianNetwork()

    variables = set()

    for parent, child in config['structure']:

        variables.add(parent)
        variables.add(child)

    for var in variables:

        values = set()

        for row in data:

            values.add(row[var])

        network.add_node(var, sorted(values))

    for parent, child in config['structure']:

        network.add_edge(parent, child)

    for var in variables:

        node = network.get_node(var)

        if not node.parents:

            cpt = build_prior(data, var)

            network.set_cpt(var, cpt)

        else:

            parent_names = [
                p.name for p in node.parents
            ]

            cpt = build_conditional_multi(
                data, var, parent_names
            )

            network.set_cpt(var, cpt)

    return network, data


def run_dataset(key, config):

    header(config['name'].upper())

    console.print(
        "\nConstruyendo red bayesiana..."
    )

    network, data = build_network(config)

    console.print(
        f"\n{network}"
    )

    print_network_ascii(network, console)

    for name in network.get_all_variables():

        node = network.get_node(name)

        is_prior = not node.parents

        console.print(
            f"\n[bold]CPT - {name}[/bold]"
        )

        print_cpt_table(
            name, node.cpt, is_prior
        )

    console.print(
        "\nInferencia probabilistica..."
    )

    engine = InferenceEngine(network)

    query_var = config['query_var']
    evidence = config['evidence']

    query_node = network.get_node(query_var)

    prior = {}

    for state in query_node.states:

        prior[state] = query_node.cpt.get(
            state, 0.0
        )

    console.print(
        f"\nEvidencia: {evidence}"
    )

    result = engine.query(query_var, evidence)

    console.print(
        f"\nP({query_var} | evidencia)"
    )

    print_query_result(query_var, result)

    console.print(
        "\nGenerando graficos en plots/..."
    )

    prefix = f"{key}_"

    plot_network_graph(
        network,
        prefix=prefix,
    )

    plot_cpt_heatmap(
        network,
        prefix=prefix,
    )

    plot_cpt_all(
        network,
        prefix=prefix,
    )

    plot_probability_comparison(
        query_var,
        prior,
        result,
        prefix=prefix,
    )

    return result


def print_usage():

    console.print(
        "Uso: python main.py [dataset]\n"
    )

    console.print(
        "Datasets disponibles:"
    )

    for key, cfg in DATASETS.items():

        console.print(
            f"  {key} - {cfg['name']}"
        )

    console.print(
        "\nSin argumento ejecuta todos."
    )


if __name__ == '__main__':

    arg = (
        sys.argv[1]
        if len(sys.argv) > 1
        else None
    )

    if arg is None:

        for key, cfg in DATASETS.items():

            run_dataset(key, cfg)

    elif arg in DATASETS:

        run_dataset(arg, DATASETS[arg])

    else:

        print_usage()
        sys.exit(1)
