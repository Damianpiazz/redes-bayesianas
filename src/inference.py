from itertools import product

from src.probability import normalize


class InferenceEngine:
    def __init__(self, network):

        self.network = network

    def enumerate_all(self, variables, evidence):

        if not variables:

            return 1.0

        first = variables[0]
        rest = variables[1:]

        node = self.network.get_node(first)

        if first in evidence:

            state = evidence[first]

            prob = self._get_probability(
                first, evidence
            )

            sub = self.enumerate_all(
                rest, evidence
            )

            return prob * sub

        total = 0.0

        for state in node.states:

            new_evidence = dict(evidence)
            new_evidence[first] = state

            prob = self._get_probability(
                first, new_evidence
            )

            sub = self.enumerate_all(
                rest, new_evidence
            )

            total += prob * sub

        return total

    def _get_probability(self, name, assignment):

        node = self.network.get_node(name)

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

        if evidence is None:

            evidence = {}

        all_vars = (
            self.network.topological_sort()
        )

        query_node = self.network.get_node(
            query_var
        )

        results = {}

        for state in query_node.states:

            full_evidence = dict(evidence)
            full_evidence[query_var] = state

            p = self.enumerate_all(
                all_vars, full_evidence
            )

            results[state] = p

        return normalize(results)

    def joint_probability(self, assignment):

        all_vars = (
            self.network.topological_sort()
        )

        return self.enumerate_all(
            all_vars, assignment
        )

    def marginal_probability(
        self, query_var, evidence=None
    ):

        result = self.query(query_var, evidence)

        return result

    def most_probable_explanation(
        self, evidence=None
    ):

        if evidence is None:

            evidence = {}

        all_vars = (
            self.network.topological_sort()
        )

        hidden = [
            v for v in all_vars
            if v not in evidence
        ]

        best_assignment = None
        best_prob = -1.0

        hidden_node_states = []

        for var in hidden:

            node = self.network.get_node(var)

            hidden_node_states.append(
                (var, node.states)
            )

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
