from src.node import Node


class BayesianNetwork:
    """Grafo acíclico dirigido (DAG) que modela una red bayesiana.

    Esta clase orquesta la estructura completa de la red: almacena los nodos
    (variables), las aristas (dependencias causales) y ofrece operaciones
    fundamentales como validación de aciclicidad, orden topológico y
    consultas sobre la estructura del grafo.

    La red se construye típicamente en tres fases:
      1. Agregar nodos con sus estados posibles.
      2. Agregar aristas que definen las relaciones padre → hijo.
      3. Asignar las CPTs a cada nodo.
    """

    def __init__(self):
        """Inicializa una red bayesiana vacía."""
        # Diccionario {nombre: Node} para acceso rápido por nombre
        self.nodes = {}
        # Lista de tuplas (padre, hijo) que registra todas las aristas
        self.edges = []

    def add_node(self, name, states):
        """Crea y registra un nuevo nodo en la red.

        Args:
            name: Nombre único de la variable.
            states: Iterable con los estados posibles de la variable.

        Returns:
            El objeto Node recién creado.
        """
        node = Node(name, states)
        self.nodes[name] = node
        return node

    def add_edge(self, parent_name, child_name):
        """Agrega una arista dirigida padre → hijo entre dos nodos.

        Actualiza las listas de padres e hijos de ambos nodos y registra
        la arista en la lista interna.

        Args:
            parent_name: Nombre del nodo padre (causa).
            child_name: Nombre del nodo hijo (efecto).

        Raises:
            KeyError: Si alguno de los nombres no existe en la red.
        """
        parent = self.nodes[parent_name]
        child = self.nodes[child_name]

        # La relación se establece en ambas direcciones para
        # facilitar la navegación del grafo en cualquiera de ellos
        parent.children.append(child)
        child.parents.append(parent)

        self.edges.append(
            (parent_name, child_name)
        )

    def set_cpt(self, name, cpt):
        """Asigna la Tabla de Probabilidad Condicional a un nodo.

        Args:
            name: Nombre del nodo.
            cpt: Diccionario con las probabilidades condicionales.
        """
        node = self.nodes[name]
        node.set_cpt(cpt)

    def get_node(self, name):
        """Obtiene un nodo por su nombre.

        Args:
            name: Nombre de la variable.

        Returns:
            El objeto Node correspondiente, o None si no existe.
        """
        return self.nodes.get(name, None)

    def get_parents(self, name):
        """Obtiene los nombres de los nodos padres de una variable.

        Args:
            name: Nombre del nodo.

        Returns:
            Lista de strings con los nombres de los padres.
        """
        node = self.nodes[name]
        return [p.name for p in node.parents]

    def get_children(self, name):
        """Obtiene los nombres de los nodos hijos de una variable.

        Args:
            name: Nombre del nodo.

        Returns:
            Lista de strings con los nombres de los hijos.
        """
        node = self.nodes[name]
        return [c.name for c in node.children]

    def get_root_nodes(self):
        """Identifica las variables raíz (sin ningún padre).

        Las variables raíz representan causas exógenas: su distribución
        de probabilidad no depende de otras variables en la red.

        Returns:
            Lista de nombres de nodos que no tienen padres.
        """
        roots = []

        for name, node in self.nodes.items():
            if not node.parents:
                roots.append(name)

        return roots

    def is_valid_dag(self):
        """Verifica que la estructura sea un grafo acíclico dirigido (DAG).

        Utiliza DFS con detección de ciclos mediante pila de recursión.
        Si se alcanza un nodo que ya está en la pila actual, existe un ciclo.

        Returns:
            True si la red es un DAG válido, False si contiene ciclos.
        """
        visited = set()        # Nodos completamente explorados
        recursion_stack = set() # Nodos en el camino de la recursión actual

        def dfs(name):
            visited.add(name)
            recursion_stack.add(name)

            node = self.nodes[name]

            for child in node.children:
                # Si el hijo está en la pila de recursión → ciclo detectado
                if child.name in recursion_stack:
                    return False

                # Explorar hijos no visitados recursivamente
                if child.name not in visited:
                    if not dfs(child.name):
                        return False

            # Backtrack: sacar el nodo de la pila al terminar su exploración
            recursion_stack.discard(name)
            return True

        # Explorar desde cada nodo no visitado (grafo podría ser disconexo)
        for name in self.nodes:
            if name not in visited:
                if not dfs(name):
                    return False

        return True

    def get_all_variables(self):
        """Obtiene la lista de todos los nombres de variables en la red.

        Returns:
            Lista de strings con los nombres de todas las variables.
        """
        return list(self.nodes.keys())

    def topological_sort(self):
        """Calcula el orden topológico de las variables de la red.

        El orden topológico garantiza que un nodo aparece después de todos
        sus padres. Es esencial para algoritmos de inferencia (como
        enumeración exhaustiva) y para la evaluación secuencial de CPTs.

        Implementación: DFS post-orden, invertido al final.

        Returns:
            Lista de nombres de variables en orden topológico.
        """
        visited = set()
        order = []  # Se llena en post-orden, se invierte después

        def dfs(name):
            visited.add(name)
            node = self.nodes[name]

            for child in node.children:
                if child.name not in visited:
                    dfs(child.name)

            # Agregar después de procesar todos los hijos (post-orden)
            order.append(name)

        for name in self.nodes:
            if name not in visited:
                dfs(name)

        # Invertir para obtener orden topológico correcto
        # (padres antes que hijos)
        order.reverse()
        return order

    def __repr__(self):
        """Representación concisa de la red para depuración."""
        n_nodes = len(self.nodes)
        n_edges = len(self.edges)

        return (
            f"BayesianNetwork("
            f"nodes={n_nodes}, "
            f"edges={n_edges})"
        )
