import csv


def load_csv(path):

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

    values = set()

    for row in data:

        values.add(row[column])

    return sorted(values)


def count_occurrences(data, column, value):

    count = 0

    for row in data:

        if row[column] == value:

            count += 1

    return count


def count_joint(
    data, col_a, val_a, col_b, val_b
):

    count = 0

    for row in data:

        if (
            row[col_a] == val_a
            and row[col_b] == val_b
        ):

            count += 1

    return count


def build_prior(data, column):

    total = len(data)
    values = get_unique_values(data, column)
    cpt = {}

    for value in values:

        count = count_occurrences(
            data, column, value
        )

        cpt[value] = count / total

    return cpt


def build_conditional(
    data, child_col, parent_col
):

    parent_values = get_unique_values(
        data, parent_col
    )

    child_values = get_unique_values(
        data, child_col
    )

    cpt = {}

    for pv in parent_values:

        parent_count = count_occurrences(
            data, parent_col, pv
        )

        for cv in child_values:

            joint = count_joint(
                data, child_col, cv,
                parent_col, pv,
            )

            key = (cv, pv)

            if parent_count > 0:

                cpt[key] = joint / parent_count

            else:

                cpt[key] = 0.0

    return cpt


def format_probability(value):

    return f"{value:.3f}"


def count_matching_rows(data, conditions):

    count = 0

    for row in data:

        match = True

        for col, val in conditions.items():

            if row[col] != val:

                match = False

                break

        if match:

            count += 1

    return count


def build_conditional_multi(
    data, child_col, parent_cols
):

    child_values = get_unique_values(
        data, child_col
    )

    parent_value_lists = []

    for pc in parent_cols:

        vals = get_unique_values(data, pc)

        parent_value_lists.append(vals)

    cpt = {}

    from itertools import product

    for parent_combo in product(
        *parent_value_lists
    ):

        conditions = {}

        for i, pc in enumerate(parent_cols):

            conditions[pc] = parent_combo[i]

        parent_count = count_matching_rows(
            data, conditions
        )

        for cv in child_values:

            full_conditions = dict(conditions)
            full_conditions[child_col] = cv

            joint = count_matching_rows(
                data, full_conditions
            )

            key = (cv,) + parent_combo

            if parent_count > 0:

                cpt[key] = joint / parent_count

            else:

                cpt[key] = 0.0

    return cpt
