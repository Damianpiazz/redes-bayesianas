from src.node import Node


class BayesianNetwork:
    def __init__(self):

        self.nodes = {}
        self.edges = []

    def add_node(self, name, states):

        node = Node(name, states)

        self.nodes[name] = node

        return node

    def add_edge(self, parent_name, child_name):

        parent = self.nodes[parent_name]
        child = self.nodes[child_name]

        parent.children.append(child)

        child.parents.append(parent)

        self.edges.append(
            (parent_name, child_name)
        )

    def set_cpt(self, name, cpt):

        node = self.nodes[name]

        node.set_cpt(cpt)

    def get_node(self, name):

        return self.nodes.get(name, None)

    def get_parents(self, name):

        node = self.nodes[name]

        return [p.name for p in node.parents]

    def get_children(self, name):

        node = self.nodes[name]

        return [c.name for c in node.children]

    def get_root_nodes(self):

        roots = []

        for name, node in self.nodes.items():

            if not node.parents:

                roots.append(name)

        return roots

    def is_valid_dag(self):

        visited = set()
        recursion_stack = set()

        def dfs(name):

            visited.add(name)
            recursion_stack.add(name)

            node = self.nodes[name]

            for child in node.children:

                if child.name in recursion_stack:

                    return False

                if child.name not in visited:

                    if not dfs(child.name):

                        return False

            recursion_stack.discard(name)

            return True

        for name in self.nodes:

            if name not in visited:

                if not dfs(name):

                    return False

        return True

    def get_all_variables(self):

        return list(self.nodes.keys())

    def topological_sort(self):

        visited = set()
        order = []

        def dfs(name):

            visited.add(name)

            node = self.nodes[name]

            for child in node.children:

                if child.name not in visited:

                    dfs(child.name)

            order.append(name)

        for name in self.nodes:

            if name not in visited:

                dfs(name)

        order.reverse()

        return order

    def __repr__(self):

        n_nodes = len(self.nodes)
        n_edges = len(self.edges)

        return (
            f"BayesianNetwork("
            f"nodes={n_nodes}, "
            f"edges={n_edges})"
        )
