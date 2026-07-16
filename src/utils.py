"""Utilidades de carga de datos y construcción de CPTs.

Este módulo ofrece funciones para:
  - Cargar datasets en formato CSV.
  - Extraer valores únicos y contar ocurrencias.
  - Construir Tablas de Probabilidad Condicional (CPTs) a partir de datos
    observados, tanto para variables raíz (prior) como para variables con
    uno o múltiples padres (condicional).
"""

import csv


def load_csv(path):
    """Carga un archivo CSV y retorna los registros como lista de diccionarios.

    Utiliza DictReader para que cada fila sea accesible por nombre de columna,
    lo que facilita el procesamiento posterior de los datos.

    Args:
        path: Ruta al archivo CSV.

    Returns:
        Lista de diccionarios, uno por fila, con claves = nombres de columna.
    """
    data = []

    with open(
        path,
        newline='',
        encoding='utf-8',
    ) as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    return data


def get_unique_values(data, column):
    """Extrae los valores únicos de una columna del dataset.

    Args:
        data: Lista de diccionarios (filas del CSV).
        column: Nombre de la columna a analizar.

    Returns:
        Lista ordenada de valores únicos.
    """
    values = set()
    for row in data:
        values.add(row[column])
    return sorted(values)


def count_occurrences(data, column, value):
    """Cuenta cuántas filas tienen un valor específico en una columna.

    Args:
        data: Lista de diccionarios.
        column: Nombre de la columna.
        value: Valor a buscar.

    Returns:
        Número de ocurrencias (int).
    """
    count = 0
    for row in data:
        if row[column] == value:
            count += 1
    return count


def count_joint(data, col_a, val_a, col_b, val_b):
    """Cuenta las filas donde dos columnas tienen valores específicos simultáneamente.

    Implementa el conteo de la probabilidad conjunta P(A=a, B=b) a partir
    de datos observados, que es la base para calcular condicionales.

    Args:
        data: Lista de diccionarios.
        col_a: Nombre de la primera columna.
        val_a: Valor esperado en la primera columna.
        col_b: Nombre de la segunda columna.
        val_b: Valor esperado en la segunda columna.

    Returns:
        Número de filas que cumplen ambas condiciones (int).
    """
    count = 0
    for row in data:
        if (
            row[col_a] == val_a
            and row[col_b] == val_b
        ):
            count += 1
    return count


def build_prior(data, column):
    """Construye la distribución marginal (prior) de una variable.

    Calcula P(X = x) para cada estado x contando la frecuencia relativa
    en los datos observados. Esta función se usa para variables raíz
    (sin padres) de la red bayesiana.

    Args:
        data: Lista de diccionarios (dataset completo).
        column: Nombre de la columna de la variable.

    Returns:
        Diccionario {estado: probabilidad} con las frecuencias relativas.
    """
    total = len(data)
    values = get_unique_values(data, column)
    cpt = {}

    for value in values:
        count = count_occurrences(data, column, value)
        cpt[value] = count / total

    return cpt


def build_conditional(data, child_col, parent_col):
    """Construye una CPT para un nodo con un único padre.

    Calcula P(hijo=hijo_val | padre=padre_val) usando frecuencias:
      P(H=h | P=p) = count(H=h, P=p) / count(P=p)

    La clave de la CPT es una tupla (valor_hijo, valor_padre).

    Args:
        data: Lista de diccionarios (dataset completo).
        child_col: Nombre de la columna del nodo hijo.
        parent_col: Nombre de la columna del nodo padre.

    Returns:
        Diccionario {(valor_hijo, valor_padre): probabilidad}.
    """
    parent_values = get_unique_values(data, parent_col)
    child_values = get_unique_values(data, child_col)
    cpt = {}

    for pv in parent_values:
        # Contar todas las filas donde el padre tiene valor pv
        parent_count = count_occurrences(
            data, parent_col, pv
        )

        for cv in child_values:
            # Contar filas donde ambos padres e hijo coinciden
            joint = count_joint(
                data, child_col, cv,
                parent_col, pv,
            )

            key = (cv, pv)

            if parent_count > 0:
                cpt[key] = joint / parent_count
            else:
                # Sin datos para esta combinación → probabilidad cero
                cpt[key] = 0.0

    return cpt


def format_probability(value):
    """Formatea un valor numérico como string con 3 decimales.

    Args:
        value: Número a formatear.

    Returns:
        String con el formato 'X.XXX'.
    """
    return f"{value:.3f}"


def count_matching_rows(data, conditions):
    """Cuenta filas que cumplen con todas las condiciones especificadas.

    Función genérica para filtrar datos con múltiples restricciones.
    Se usa como base para construir CPTs con múltiples padres.

    Args:
        data: Lista de diccionarios.
        conditions: Diccionario {columna: valor} con las condiciones a cumplir.

    Returns:
        Número de filas que cumplen todas las condiciones (int).
    """
    count = 0
    for row in data:
        match = True
        for col, val in conditions.items():
            if row[col] != val:
                match = False
                break  # Optimización: salir en la primera discrepancia

        if match:
            count += 1

    return count


def build_conditional_multi(data, child_col, parent_cols):
    """Construye una CPT para un nodo con múltiples padres.

    Generaliza build_conditional() a N padres. Para cada combinación
    de valores de los padres, calcula P(hijo | padre1, padre2, ...).

    La clave de la CPT es una tupla (valor_hijo, val_padre1, val_padre2, ...),
    donde el orden de los padres debe coincidir con el de parent_cols.

    La complejidad crece exponencialmente con el número de padres,
    ya que se evalúa el producto cartesiano de todos sus estados.

    Args:
        data: Lista de diccionarios (dataset completo).
        child_col: Nombre de la columna del nodo hijo.
        parent_cols: Lista de nombres de columnas de los nodos padres.

    Returns:
        Diccionario {(valor_hijo, val_padre1, ...): probabilidad}.
    """
    child_values = get_unique_values(data, child_col)

    # Recolectar los estados posibles de cada padre
    parent_value_lists = []
    for pc in parent_cols:
        vals = get_unique_values(data, pc)
        parent_value_lists.append(vals)

    cpt = {}

    from itertools import product

    # Iterar sobre el producto cartesiano de todos los estados de padres
    for parent_combo in product(*parent_value_lists):
        # Construir las condiciones de filtrado para esta combinación
        conditions = {}
        for i, pc in enumerate(parent_cols):
            conditions[pc] = parent_combo[i]

        # Contar filas donde todos los padres toman sus valores correspondientes
        parent_count = count_matching_rows(
            data, conditions
        )

        for cv in child_values:
            # Agregar la condición del hijo a las condiciones de padres
            full_conditions = dict(conditions)
            full_conditions[child_col] = cv

            # Contar la conjunta: (padres en valores específicos) AND (hijo = cv)
            joint = count_matching_rows(
                data, full_conditions
            )

            # Clave: (hijo, padre1, padre2, ...)
            key = (cv,) + parent_combo

            if parent_count > 0:
                cpt[key] = joint / parent_count
            else:
                cpt[key] = 0.0

    return cpt
