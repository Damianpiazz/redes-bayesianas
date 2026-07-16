"""Módulo de visualización para redes bayesianas.

Proporciona funciones para generar gráficos que ayudan a entender
la estructura de la red y las distribuciones de probabilidad:

  - Grafo de la red (diagrama dirigido con nodos coloreados por tipo).
  - Heatmaps de las CPTs individuales (prior y condicional).
  - Panel combinado de todas las CPTs.
  - Comparación visual prior vs. posterior después de la inferencia.
  - Impresión ASCII de la estructura para terminal.

Se usa matplotlib con backend 'Agg' (sin display) para generar
archivos PNG en el directorio 'plots/'.
"""

import os

import matplotlib

matplotlib.use('Agg')  # Backend no interactivo, solo para guardar archivos

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import networkx as nx


def ensure_plots_dir():
    """Asegura que el directorio 'plots/' exista para guardar gráficos.

    Crea el directorio de forma recursiva si no existe. Utiliza
    exist_ok=True para evitar errores si ya está creado.
    """
    os.makedirs(
        'plots',
        exist_ok=True,
    )


def plot_network_graph(network, prefix=''):
    """Genera un gráfico del grafo de la red bayesiana usando NetworkX.

    Los nodos raíz (sin padres) se muestran en verde turquesa y los
    nodos hijo (con padres) en rojo, lo que permite identificar
    visualmente las causas exógenas de la red.

    Args:
        network: Instancia de BayesianNetwork.
        prefix: Prefijo opcional para el nombre del archivo de salida.
    """
    ensure_plots_dir()

    # Construir el grafo dirigido de NetworkX
    G = nx.DiGraph()

    for name in network.get_all_variables():
        G.add_node(name)

    for parent, child in network.edges:
        G.add_edge(parent, child)

    fig, ax = plt.subplots(
        figsize=(8, 6),
    )

    # Layout con semilla fija para reproducibilidad
    pos = nx.spring_layout(
        G,
        seed=42,
        k=2.0,  # Distancia de separación entre nodos
    )

    root_nodes = network.get_root_nodes()

    # Colorear nodos: verde turquesa = raíz, rojo = hijo
    node_colors = []
    for node in G.nodes():
        if node in root_nodes:
            node_colors.append('#4ECDC4')
        else:
            node_colors.append('#FF6B6B')

    # Dibujar aristas con estilo curvo para mejor legibilidad
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color='#555555',
        width=2.0,
        arrows=True,
        arrowsize=20,
        arrowstyle='-|>',
        connectionstyle='arc3,rad=0.1',  # Curvatura sutil
        ax=ax,
    )

    # Dibujar nodos con borde negro y tamaño grande para legibilidad
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=1500,
        edgecolors='black',
        linewidths=2.0,
        ax=ax,
    )

    # Etiquetas de los nodos en blanco sobre fondo de color
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=10,
        font_weight='bold',
        font_color='white',
        ax=ax,
    )

    # Leyenda para distinguir tipos de nodo
    handles = [
        mpatches.Patch(
            color='#4ECDC4',
            label='Raiz (sin padres)',
        ),
        mpatches.Patch(
            color='#FF6B6B',
            label='Hijo (con padres)',
        ),
    ]

    ax.legend(
        handles=handles,
        title='Tipo de nodo',
        loc='upper right',
        fontsize=9,
    )

    ax.set_title(
        'Grafo de la Red Bayesiana',
        fontsize=14,
        fontweight='bold',
    )

    ax.set_axis_off()
    plt.tight_layout()

    plt.savefig(
        f'plots/{prefix}network_graph.png',
        dpi=150,
    )
    plt.close()


def plot_cpt_heatmap(network, prefix=''):
    """Genera heatmaps individuales para cada CPT de la red.

    Para nodos raíz genera un heatmap unidimensional (prior).
    Para nodos con padres genera un heatmap 2D donde las filas
    representan combinaciones de valores de padres y las columnas
    los estados del hijo.

    Args:
        network: Instancia de BayesianNetwork.
        prefix: Prefijo para los nombres de archivo.
    """
    ensure_plots_dir()

    for name in network.get_all_variables():
        node = network.get_node(name)

        if not node.parents:
            _plot_prior_heatmap(node, prefix)
        else:
            _plot_conditional_heatmap(node, prefix)


def _plot_prior_heatmap(node, prefix=''):
    """Genera un heatmap para la distribución prior de un nodo raíz.

    Muestra una fila única con las probabilidades de cada estado,
    usando escala de color de amarillo (baja) a rojo (alta).

    Args:
        node: Nodo raíz (sin padres) de la red.
        prefix: Prefijo para el nombre del archivo de salida.
    """
    states = node.states
    probs = [
        node.cpt.get(s, 0.0)
        for s in states
    ]

    # Reshape a matriz 1×N para seaborn.heatmap
    data = np.array(probs).reshape(1, -1)

    plt.figure(figsize=(6, 2))

    sns.heatmap(
        data,
        annot=True,       # Mostrar valores numéricos en cada celda
        fmt='.3f',         # Formato con 3 decimales
        cmap='YlOrRd',    # Paleta de colores: amarillo → naranja → rojo
        xticklabels=states,
        yticklabels=[node.name],
        linewidths=0.5,
        cbar_kws={
            'label': 'Probabilidad',
        },
    )

    plt.title(
        f'P({node.name}) - Probabilidad Prior',
        fontsize=12,
        fontweight='bold',
    )

    plt.tight_layout()
    plt.savefig(
        f'plots/{prefix}cpt_{node.name}.png',
        dpi=150,
    )
    plt.close()


def _plot_conditional_heatmap(node, prefix=''):
    """Genera un heatmap 2D para la CPT de un nodo con padres.

    Las filas representan las combinaciones de valores de los padres
    y las columnas los estados del nodo hijo. Cada celda contiene
    la probabilidad condicional correspondiente.

    Para múltiples padres, las filas muestran combinaciones tipo
    "valor_padre1 / valor_padre2".

    Args:
        node: Nodo con al menos un padre.
        prefix: Prefijo para el nombre del archivo de salida.
    """
    parent_names = [
        p.name for p in node.parents
    ]

    child_states = node.states

    # Recolectar estados de cada padre para generar combinaciones
    parent_state_lists = []
    for p in node.parents:
        parent_state_lists.append(p.states)

    from itertools import product

    # Producto cartesiano de todos los estados de los padres
    parent_combos = list(
        product(*parent_state_lists)
    )

    n_rows = len(parent_combos)
    n_cols = len(child_states)

    # Construir la matriz de probabilidades
    data = np.zeros((n_rows, n_cols))

    for i, combo in enumerate(parent_combos):
        for j, cs in enumerate(child_states):
            # Clave CPT: (estado_hijo, val_padre1, val_padre2, ...)
            key = (cs,) + combo
            data[i][j] = node.cpt.get(key, 0.0)

    # Etiquetas de filas: combinaciones de padres separadas por '/'
    row_labels = [
        ' / '.join(combo)
        for combo in parent_combos
    ]

    # Tamaño del gráfico proporcional a la cantidad de combinaciones
    plt.figure(
        figsize=(
            max(6, n_cols * 1.5),
            max(3, n_rows * 0.8),
        ),
    )

    sns.heatmap(
        data,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        xticklabels=child_states,
        yticklabels=row_labels,
        linewidths=0.5,
        cbar_kws={
            'label': 'Probabilidad',
        },
    )

    parent_str = ', '.join(parent_names)

    plt.title(
        f'P({node.name} | {parent_str})',
        fontsize=12,
        fontweight='bold',
        )
    plt.xlabel(node.name)
    plt.ylabel('Valores padres')

    plt.tight_layout()
    plt.savefig(
        f'plots/{prefix}cpt_{node.name}.png',
        dpi=150,
    )
    plt.close()


def plot_probability_comparison(
    query_var,
    prior,
    posterior,
    prefix='',
):
    """Genera un gráfico de barras comparando prior vs. posterior.

    Visualiza el efecto de la evidencia en la distribución de probabilidad.
    Las barras verdes muestran la distribución previa (sin evidencia) y
    las rojas la distribución actualizada (con evidencia).

    Args:
        query_var: Nombre de la variable de consulta.
        prior: Diccionario {estado: probabilidad} sin evidencia.
        posterior: Diccionario {estado: probabilidad} con evidencia.
        prefix: Prefijo para el nombre del archivo de salida.
    """
    ensure_plots_dir()

    states = list(prior.keys())

    prior_vals = [prior[s] for s in states]
    posterior_vals = [posterior[s] for s in states]

    x = np.arange(len(states))
    width = 0.35  # Ancho de cada barra

    fig, ax = plt.subplots(figsize=(8, 5))

    # Barras agrupadas lado a lado para comparación visual directa
    bars1 = ax.bar(
        x - width / 2,
        prior_vals,
        width,
        label='Prior (sin evidencia)',
        color='#4ECDC4',
        edgecolor='black',
        linewidth=0.5,
    )

    bars2 = ax.bar(
        x + width / 2,
        posterior_vals,
        width,
        label='Posterior (con evidencia)',
        color='#FF6B6B',
        edgecolor='black',
        linewidth=0.5,
    )

    # Etiquetas numéricas encima de cada barra
    ax.bar_label(
        bars1,
        fmt='%.3f',
        padding=3,
        fontsize=9,
    )
    ax.bar_label(
        bars2,
        fmt='%.3f',
        padding=3,
        fontsize=9,
    )

    ax.set_xlabel('Estados', fontsize=11)
    ax.set_ylabel('Probabilidad', fontsize=11)

    ax.set_title(
        f'P({query_var}): Prior vs Posterior',
        fontsize=14,
        fontweight='bold',
    )

    ax.set_xticks(x)
    ax.set_xticklabels(states)
    ax.set_ylim(0, 1.15)  # Espacio extra para las etiquetas encima
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        f'plots/{prefix}comparison_{query_var}.png',
        dpi=150,
    )
    plt.close()


def plot_cpt_all(network, prefix=''):
    """Genera un panel combinado con todas las CPTs de la red en una sola imagen.

    Utiliza un grid de subplots donde cada variable ocupa una celda.
    Los nodos raíz se muestran como gráficos de barras y los nodos
    con padres como heatmaps. Las celdas vacías se ocultan.

    Args:
        network: Instancia de BayesianNetwork.
        prefix: Prefijo para el nombre del archivo de salida.
    """
    ensure_plots_dir()

    variables = network.topological_sort()
    n = len(variables)
    cols = 2
    rows = (n + cols - 1) // cols  # Redondeo hacia arriba

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 6, rows * 4),
    )

    # Manejar el caso de una sola variable (axes no es array)
    if n == 1:
        axes = np.array([axes])

    axes_flat = axes.flatten()

    for i, name in enumerate(variables):
        node = network.get_node(name)

        if not node.parents:
            # ─── Nodo raíz: gráfico de barras con probabilidades prior ───
            states = node.states
            probs = [
                node.cpt.get(s, 0.0)
                for s in states
            ]

            ax = axes_flat[i]

            sns.barplot(
                x=states,
                y=probs,
                hue=states,
                palette='Set2',
                dodge=False,
                legend=False,
                ax=ax,
                edgecolor='black',
                linewidth=0.5,
            )

            ax.set_title(
                f'P({name})',
                fontsize=11,
                fontweight='bold',
            )
            ax.set_ylim(0, 1.1)

            # Etiquetas numéricas encima de cada barra
            for j, v in enumerate(probs):
                ax.text(
                    j,
                    v + 0.03,
                    f'{v:.3f}',
                    ha='center',
                    fontsize=9,
                )
        else:
            # ─── Nodo con padres: heatmap de la CPT ───
            parent_names = [
                p.name for p in node.parents
            ]

            parent_state_lists = [
                p.states
                for p in node.parents
            ]

            from itertools import product

            combos = list(
                product(*parent_state_lists)
            )

            child_states = node.states
            n_rows_cpt = len(combos)
            n_cols_cpt = len(child_states)

            data = np.zeros(
                (n_rows_cpt, n_cols_cpt)
            )

            for ri, combo in enumerate(combos):
                for ci, cs in enumerate(child_states):
                    key = (cs,) + combo
                    data[ri][ci] = (
                        node.cpt.get(key, 0.0)
                    )

            ax = axes_flat[i]

            row_labels = [
                ' / '.join(c)
                for c in combos
            ]

            sns.heatmap(
                data,
                annot=True,
                fmt='.3f',
                cmap='YlOrRd',
                xticklabels=child_states,
                yticklabels=row_labels,
                linewidths=0.3,
                ax=ax,
                cbar_kws={
                    'shrink': 0.8,
                },
            )

            parent_str = ', '.join(parent_names)

            ax.set_title(
                f'P({name} | {parent_str})',
                fontsize=11,
                fontweight='bold',
            )

    # Ocultar celdas vacías si el número de variables no llena el grid
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle(
        'Tablas de Probabilidad Condicional',
        fontsize=14,
        fontweight='bold',
        y=1.01,
    )

    plt.tight_layout()
    plt.savefig(
        f'plots/{prefix}cpt_all.png',
        dpi=150,
        bbox_inches='tight',
    )
    plt.close()


def print_network_ascii(network, console):
    """Imprime la estructura del grafo en formato ASCII en la terminal.

    Muestra la jerarquía de la red en orden topológico, indicando
    cuáles son nodos raíz y cuáles son hijos de otros nodos.

    Args:
        network: Instancia de BayesianNetwork.
        console: Instancia de rich.console.Console para impresión formateada.
    """
    console.print(
        "\n[bold]Estructura del grafo:[/bold]"
    )

    for name in network.topological_sort():
        node = network.get_node(name)

        if not node.parents:
            console.print(
                f"  {name} (raiz)"
            )
        else:
            # Formato: "padre1, padre2 -> hijo"
            parents = ', '.join(
                p.name for p in node.parents
            )
            console.print(
                f"  {parents} -> {name}"
            )
