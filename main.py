"""Punto de entrada principal para la demostración de redes bayesianas.

Este script construye una red bayesiana a partir de un dataset CSV,
calcula las CPTs desde los datos, realiza inferencia probabilística
usando enumeración exacta y genera gráficos de visualización.

Uso:
    python main.py           # Ejecuta todos los datasets configurados
    python main.py weather   # Ejecuta solo el dataset 'weather'
"""

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


# Consola rich para salida formateada con colores y tablas
console = Console()


# ─── Configuración de datasets ────────────────────────────────────────
# Cada dataset define: nombre descriptivo, ruta al CSV, variable de
# consulta, evidencia observada y la estructura de la red (aristas).
DATASETS = {
    'weather': {
        'name': 'Weather (Rain, Sprinkler, Wet Grass)',
        'path': 'data/weather.csv',
        'query_var': 'Lluvia',
        'evidence': {'Cesped_Mojado': 'Si'},
        # Estructura causal: Lluvia causa Aspersor y Cesped_Mojado,
        # y Aspersor también causa Cesped_Mojado (vía conjunta)
        'structure': [
            ('Lluvia', 'Aspersor'),
            ('Lluvia', 'Cesped_Mojado'),
            ('Aspersor', 'Cesped_Mojado'),
        ],
    },
}


def header(text):
    """Imprime un encabezado con formato en la terminal.

    Args:
        text: Texto del encabezado a mostrar.
    """
    console.print(f"\n[bold]{text}[/bold]")
    console.print("-" * len(text))


def print_cpt_table(name, cpt, is_prior):
    """Imprime una CPT como tabla formateada en la terminal.

    Adapta el formato según si es una distribución prior (simple) o
    condicional (con columnas de padres).

    Args:
        name: Nombre de la variable.
        cpt: Diccionario con la Tabla de Probabilidad Condicional.
        is_prior: True si el nodo es raíz (sin padres).
    """
    table = Table(show_header=True)

    if is_prior:
        # Distribución prior: solo estado y probabilidad
        table.add_column(name.upper())
        table.add_column("PROBABILIDAD", justify="right")

        for state, prob in cpt.items():
            table.add_row(
                state,
                format_probability(prob),
            )
    else:
        # Distribución condicional: columnas dinámicas para padres
        sample_key = next(iter(cpt))
        n_parents = len(sample_key) - 1

        for i in range(n_parents):
            table.add_column(f"PADRE_{i+1}")

        table.add_column(name.upper())
        table.add_column("P(CONDICIONAL)", justify="right")

        for key, prob in cpt.items():
            child_state = key[0]
            parent_states = key[1:]

            # Reconstruir la fila: padres primero, luego hijo y probabilidad
            row = list(parent_states) + [
                child_state,
                format_probability(prob),
            ]
            table.add_row(*row)

    console.print(table)


def print_query_result(query_var, result):
    """Imprime el resultado de una consulta de inferencia como tabla.

    Args:
        query_var: Nombre de la variable consultada.
        result: Diccionario {estado: probabilidad} normalizado.
    """
    table = Table(show_header=True)

    table.add_column(f"P({query_var})")
    table.add_column("PROBABILIDAD", justify="right")

    for state, prob in result.items():
        table.add_row(
            state,
            format_probability(prob),
        )

    console.print(table)


def build_network(config):
    """Construye una red bayesiana completa desde una configuración.

    Proceso:
      1. Carga los datos del CSV.
      2. Extrae las variables de la estructura de aristas.
      3. Crea nodos con sus estados (extraídos de los datos).
      4. Agrega las aristas definidas en la configuración.
      5. Construye las CPTs calculando frecuencias de los datos:
         - Nodos raíz → build_prior()
         - Nodos con padres → build_conditional_multi()

    Args:
        config: Diccionario de configuración del dataset (ver DATASETS).

    Returns:
        Tupla (BayesianNetwork, lista de datos CSV).
    """
    data = load_csv(config['path'])

    network = BayesianNetwork()

    # Recolectar todas las variables únicas de la estructura de aristas
    variables = set()
    for parent, child in config['structure']:
        variables.add(parent)
        variables.add(child)

    # Crear nodos con sus estados posibles extraídos de los datos
    for var in variables:
        values = set()
        for row in data:
            values.add(row[var])
        network.add_node(var, sorted(values))

    # Agregar las aristas de dependencia causal
    for parent, child in config['structure']:
        network.add_edge(parent, child)

    # Calcular y asignar las CPTs desde los datos observados
    for var in variables:
        node = network.get_node(var)

        if not node.parents:
            # Nodo raíz: distribución marginal (prior)
            cpt = build_prior(data, var)
            network.set_cpt(var, cpt)
        else:
            # Nodo con padres: distribución condicional
            parent_names = [
                p.name for p in node.parents
            ]
            cpt = build_conditional_multi(
                data, var, parent_names
            )
            network.set_cpt(var, cpt)

    return network, data


def run_dataset(key, config):
    """Ejecuta el pipeline completo para un dataset: construcción, inferencia y visualización.

    Flujo de ejecución:
      1. Construir la red bayesiana desde los datos.
      2. Mostrar la estructura del grafo.
      3. Mostrar todas las CPTs.
      4. Ejecutar inferencia P(query_var | evidence).
      5. Generar gráficos de visualización.

    Args:
        key: Clave del dataset en DATASETS (usada como prefijo de archivos).
        config: Diccionario de configuración del dataset.

    Returns:
        Diccionario con la distribución posterior resultante.
    """
    header(config['name'].upper())

    console.print("\nConstruyendo red bayesiana...")
    network, data = build_network(config)

    console.print(f"\n{network}")
    print_network_ascii(network, console)

    # Mostrar cada CPT en la terminal
    for name in network.get_all_variables():
        node = network.get_node(name)
        is_prior = not node.parents

        console.print(f"\n[bold]CPT - {name}[/bold]")
        print_cpt_table(name, node.cpt, is_prior)

    # Ejecutar inferencia probabilística
    console.print("\nInferencia probabilistica...")
    engine = InferenceEngine(network)

    query_var = config['query_var']
    evidence = config['evidence']

    # Calcular la distribución prior de la variable de consulta
    # para comparar con el posterior después de la inferencia
    query_node = network.get_node(query_var)
    prior = {}
    for state in query_node.states:
        prior[state] = query_node.cpt.get(state, 0.0)

    console.print(f"\nEvidencia: {evidence}")

    # Consulta principal: P(query_var | evidence)
    result = engine.query(query_var, evidence)

    console.print(f"\nP({query_var} | evidencia)")
    print_query_result(query_var, result)

    # Generar todos los gráficos de visualización
    console.print("\nGenerando graficos en plots/...")
    prefix = f"{key}_"

    plot_network_graph(network, prefix=prefix)
    plot_cpt_heatmap(network, prefix=prefix)
    plot_cpt_all(network, prefix=prefix)
    plot_probability_comparison(
        query_var,
        prior,
        result,
        prefix=prefix,
    )

    return result


def print_usage():
    """Muestra instrucciones de uso y los datasets disponibles."""
    console.print(
        "Uso: python main.py [dataset]\n"
    )
    console.print("Datasets disponibles:")

    for key, cfg in DATASETS.items():
        console.print(f"  {key} - {cfg['name']}")

    console.print("\nSin argumento ejecuta todos.")


if __name__ == '__main__':
    # Parseo simple de argumentos: nombre del dataset o ninguno (todos)
    arg = (
        sys.argv[1]
        if len(sys.argv) > 1
        else None
    )

    if arg is None:
        # Sin argumentos: ejecutar todos los datasets configurados
        for key, cfg in DATASETS.items():
            run_dataset(key, cfg)
    elif arg in DATASETS:
        # Dataset específico solicitado
        run_dataset(arg, DATASETS[arg])
    else:
        print_usage()
        sys.exit(1)
