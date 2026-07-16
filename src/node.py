class Node:
    def __init__(self, name, states, parents=None):

        self.name = name
        self.states = list(states)
        self.parents = list(parents) if parents else []
        self.children = []
        self.cpt = {}

    def set_cpt(self, cpt):

        self.cpt = cpt

    def get_probability(self, state, parent_values=None):

        if not self.parents:

            return self.cpt.get(state, 0.0)

        key = self._make_key(state, parent_values)

        return self.cpt.get(key, 0.0)

    def _make_key(self, state, parent_values):

        if parent_values is None:

            return state

        parts = [state]

        for parent in self.parents:

            val = parent_values.get(parent.name, None)

            parts.append(val)

        return tuple(parts)

    def get_prior(self):

        if self.parents:

            return None

        return dict(self.cpt)

    def get_conditional(self):

        if not self.parents:

            return None

        return dict(self.cpt)

    def __repr__(self):

        n_parents = len(self.parents)

        return (
            f"Node({self.name}, "
            f"states={len(self.states)}, "
            f"parents={n_parents})"
        )
