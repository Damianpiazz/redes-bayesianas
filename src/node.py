class Node:
    """Representa una variable (nodo) dentro de una red bayesiana.

    Cada nodo encapsula un nombre descriptivo, los estados posibles que
    puede adoptar, sus relaciones de parentesco con otros nodos y su
    Tabla de Probabilidad Condicional (CPT).

    La CPT almacena las probabilidades de transición del nodo:
      - Para nodos raíz (sin padres): P(estado) como mapeo directo.
      - Para nodos con padres: P(estado | valores_padres) usando claves tuple.

    La forma de las claves de la CPT se define internamente y debe ser
    consistente con la utilizada por el motor de inferencia.
    """

    def __init__(self, name, states, parents=None):
        """Inicializa un nodo de la red bayesiana.

        Args:
            name: Nombre único de la variable (ej. 'Lluvia', 'Aspersor').
            states: Iterable con los valores que puede tomar la variable
                    (ej. ['Si', 'No']).
            parents: Lista opcional de nodos padre. Si se omite, el nodo
                     se considera una variable raíz de la red.
        """
        self.name = name
        self.states = list(states)
        self.parents = list(parents) if parents else []
        self.children = []  # Se completa cuando se agregan aristas
        self.cpt = {}       # Tabla de Probabilidad Condicional

    def set_cpt(self, cpt):
        """Asigna la Tabla de Probabilidad Condicional (CPT) al nodo.

        Args:
            cpt: Diccionario con las probabilidades. Para nodos raíz es
                 {estado: probabilidad}. Para nodos con padres es
                 {(estado_hijo, val_padre1, ...): probabilidad}.
        """
        self.cpt = cpt

    def get_probability(self, state, parent_values=None):
        """Obtiene la probabilidad de un estado dado el contexto de padres.

        Args:
            state: Estado del nodo cuya probabilidad se busca.
            parent_values: Diccionario {nombre_padre: valor} con los valores
                           asignados a los padres. None si el nodo es raíz.

        Returns:
            Probabilidad del estado dado el contexto, o 0.0 si la clave
            no existe en la CPT.
        """
        # Nodos raíz: la probabilidad depende únicamente del estado
        if not self.parents:
            return self.cpt.get(state, 0.0)

        # Nodos con padres: se construye una clave compuesta
        key = self._make_key(state, parent_values)
        return self.cpt.get(key, 0.0)

    def _make_key(self, state, parent_values):
        """Construye la clave tuple para acceder a la CPT.

        La clave tiene la forma (estado_hijo, valor_padre1, valor_padre2, ...).
        El orden de los padres es el mismo en el que se registraron,
        lo cual debe ser consistente con la construcción de la CPT.

        Args:
            state: Estado del nodo hijo.
            parent_values: Diccionario con los valores de los padres.

        Returns:
            Tupla con la clave compuesta para la CPT.
        """
        if parent_values is None:
            return state

        parts = [state]

        for parent in self.parents:
            # Se extrae el valor del padre, o None si no está asignado
            val = parent_values.get(parent.name, None)
            parts.append(val)

        return tuple(parts)

    def get_prior(self):
        """Obtiene la distribución de probabilidad marginal (prior).

        Solo tiene sentido para nodos raíz (sin padres), ya que representan
        variables independientes.

        Returns:
            Diccionario {estado: probabilidad} o None si el nodo tiene padres.
        """
        if self.parents:
            return None

        return dict(self.cpt)

    def get_conditional(self):
        """Obtiene la tabla completa de probabilidades condicionales.

        Solo tiene sentido para nodos con al menos un padre.

        Returns:
            Diccionario con las probabilidades condicionales, o None si
            el nodo es raíz (no tiene dependencias condicionales).
        """
        if not self.parents:
            return None

        return dict(self.cpt)

    def __repr__(self):
        """Representación concisa del nodo para depuración."""
        n_parents = len(self.parents)

        return (
            f"Node({self.name}, "
            f"states={len(self.states)}, "
            f"parents={n_parents})"
        )
