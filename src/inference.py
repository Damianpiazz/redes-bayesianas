"""Motor de inferencia para redes bayesianas por enumeración exhaustiva.

Implementa el algoritmo de enumeración exacta descrito en Russell & Norvig.
Para cada consulta P(X | evidencia), el algoritmo enumera todas las
asignaciones posibles de las variables ocultas, acumula las probabilidades
y normaliza el resultado.

La complejidad es O(n · |X|^(n-m)) donde n es el número de variables
y m el de variables con evidencia observable.
"""

from itertools import product

from src.probability import normalize


class InferenceEngine:
    """Motor de inferencia que realiza consultas exactas sobre una red bayesiana.

    Utiliza enumeración exhaustiva sobre el orden topológico de la red.
    Para cada consulta, itera sobre todos los estados posibles de las
    variables de:hidden, calcula la probabilidad conjunta y normaliza.
    """

    def __init__(self, network):
        """Inicializa el motor con una red bayesiana ya construida.

        Args:
            network: Instancia de BayesianNetwork con nodos y CPTs configurados.
        """
        self.network = network

    def enumerate_all(self, variables, evidence):
        """Calcula la suma de probabilidades sobre todas las asignaciones posibles.

        Implementa la recurrencia central del algoritmo de enumeración.
        Procesa recursivamente cada variable:
          - Si la variable tiene evidencia fija, multiplica su probabilidad
            y continúa con las siguientes.
          - Si es una variable oculta, suma sobre todos sus estados posibles.

        La recursion bottom case (sin variables) retorna 1.0, que actúa
        como identidad multiplicativa para la cadena de factores.

        Args:
            variables: Lista de nombres de variables en orden topológico.
            evidence: Diccionario {variable: estado} con la evidencia conocida.

        Returns:
            Probabilidad no normalizada de la asignación completa.
        """
        # Caso base: no quedan variables por procesar
        if not variables:
            return 1.0

        first = variables[0]
        rest = variables[1:]
        node = self.network.get_node(first)

        # Si la variable tiene evidencia conocida, no se suma sobre sus estados
        if first in evidence:
            state = evidence[first]
            prob = self._get_probability(first, evidence)
            sub = self.enumerate_all(rest, evidence)
            return prob * sub

        # Variable oculta: sumar sobre todos sus estados posibles
        total = 0.0

        for state in node.states:
            # Extender la evidencia con cada estado candidato
            new_evidence = dict(evidence)
            new_evidence[first] = state

            prob = self._get_probability(first, new_evidence)
            sub = self.enumerate_all(rest, new_evidence)
            total += prob * sub

        return total

    def _get_probability(self, name, assignment):
        """Obtiene P(estado | valores_padres) para un nodo dado.

        Extrae los valores de los padres desde la asignación completa
        y consulta la CPT del nodo.

        Args:
            name: Nombre del nodo.
            assignment: Diccionario con la asignación actual de todas las variables.

        Returns:
            Probabilidad del estado actual del nodo dado el contexto de padres.
        """
        node = self.network.get_node(name)

        # Extraer solo los valores de los padres (necesarios para la CPT)
        parent_values = {}
        for parent in node.parents:
            parent_values[parent.name] = (
                assignment[parent.name]
            )

        current_state = assignment[name]

        return node.get_probability(
            current_state, parent_values
        )

    def query(self, query_var, evidence=None):
        """Calcula la distribución P(query_var | evidencia).

        Para cada estado de la variable de consulta, fija ese estado
        junto con la evidencia proporcionada, calcula la probabilidad
        conjunta usando enumeración, y normaliza el resultado final.

        Args:
            query_var: Nombre de la variable sobre la que se consulta.
            evidence: Diccionario opcional {variable: estado} con evidencia.

        Returns:
            Diccionario {estado: probabilidad} normalizado.
        """
        if evidence is None:
            evidence = {}

        # El orden topológico garantiza que los padres se procesan antes
        all_vars = self.network.topological_sort()
        query_node = self.network.get_node(query_var)
        results = {}

        for state in query_node.states:
            # Fijar el estado de la variable de consulta como evidencia temporal
            full_evidence = dict(evidence)
            full_evidence[query_var] = state

            p = self.enumerate_all(all_vars, full_evidence)
            results[state] = p

        return normalize(results)

    def joint_probability(self, assignment):
        """Calcula P(asignación completa) = P(X1=x1, X2=x2, ..., Xn=xn).

        Utiliza la factorización de la distribución conjunta definida por
        la estructura del DAG:
          P(X1, ..., Xn) = Π_i P(Xi | padres(Xi))

        Args:
            assignment: Diccionario {variable: estado} con la asignación completa.

        Returns:
            Probabilidad conjunta (float).
        """
        all_vars = self.network.topological_sort()

        return self.enumerate_all(all_vars, assignment)

    def marginal_probability(self, query_var, evidence=None):
        """Calcula la probabilidad marginal P(query_var | evidence).

        Función wrapper de query() que retorna la distribución completa.
        Útil cuando se necesita la distribución en lugar de un valor escalar.

        Args:
            query_var: Nombre de la variable de consulta.
            evidence: Diccionario opcional con evidencia.

        Returns:
            Diccionario {estado: probabilidad} normalizado.
        """
        result = self.query(query_var, evidence)
        return result

    def most_probable_explanation(self, evidence=None):
        """Encuentra la explicación más probable (MPE) dada la evidencia.

        El problema MPE busca la asignación de todas las variables ocultas
        que maximiza P(asignación completa | evidencia). Esto equivale a
        encontrar la distribución conjunta de mayor masa.

        Implementación: fuerza bruta sobre el producto cartesiano de todos
        los estados de las variables ocultas. Adecuado para redes pequeñas.

        Args:
            evidence: Diccionario {variable: estado} con la evidencia conocida.

        Returns:
            Tupla (mejor_asignación, mejor_probabilidad) donde
            mejor_asignación es un dict {variable: estado}.
        """
        if evidence is None:
            evidence = {}

        all_vars = self.network.topological_sort()

        # Variables sin evidencia → son las que hay que explicar
        hidden = [
            v for v in all_vars
            if v not in evidence
        ]

        best_assignment = None
        best_prob = -1.0

        # Preparar los estados posibles de cada variable oculta
        hidden_node_states = []
        for var in hidden:
            node = self.network.get_node(var)
            hidden_node_states.append(
                (var, node.states)
            )

        # Evaluar cada combinación posible de estados para las variables ocultas
        for combo in product(
            *[states
              for _, states in hidden_node_states]
        ):
            full = dict(evidence)

            for i, (var, _) in enumerate(
                hidden_node_states
            ):
                full[var] = combo[i]

            p = self.joint_probability(full)

            if p > best_prob:
                best_prob = p
                best_assignment = dict(full)

        return best_assignment, best_prob
