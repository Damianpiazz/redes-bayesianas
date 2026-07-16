"""Utilidades de cálculo probabilístico para redes bayesianas.

Este módulo proporciona funciones fundamentales para el manejo de
probabilidades: validación, cálculo de condicionales, conjuntas,
marginales, normalización y el teorema de Bayes.
"""


def validate_probability(p):
    """Valida que un valor sea una probabilidad válida en el rango [0, 1].

    Args:
        p: Valor numérico a validar.

    Returns:
        True si el valor es válido.

    Raises:
        TypeError: Si p no es numérico (int o float).
        ValueError: Si p está fuera del rango [0, 1].
    """
    if not isinstance(p, (int, float)):
        raise TypeError(
            f"Probability must be numeric, got {type(p)}"
        )

    if p < 0.0 or p > 1.0:
        raise ValueError(
            f"Probability must be in [0, 1], got {p}"
        )

    return True


def conditional_probability(p_joint, p_evidence):
    """Calcula P(A | B) = P(A ∩ B) / P(B).

    Implementa la definición clásica de probabilidad condicional.
    Requiere que P(B) > 0 para evitar división por cero.

    Args:
        p_joint: P(A ∩ B) — probabilidad conjunta.
        p_evidence: P(B) — probabilidad del evento condicionante.

    Returns:
        P(A | B) — probabilidad condicional.

    Raises:
        ValueError: Si p_evidence es cero (división por cero).
    """
    validate_probability(p_joint)
    validate_probability(p_evidence)

    if p_evidence == 0.0:
        raise ValueError(
            "P(evidence) cannot be zero"
        )

    return p_joint / p_evidence


def joint_probability(*probabilities):
    """Calcula la probabilidad conjunta de eventos independientes.

    Asume independencia entre los eventos y multiplica sus probabilidades:
    P(A ∩ B ∩ ... ∩ N) = P(A) × P(B) × ... × P(N).

    Nota: En una red bayesiana real, la factorización de la conjunta
    se basa en la estructura del DAG, no en independencia absoluta.

    Args:
        *probabilities: Cantidad variable de probabilidades a multiplicar.

    Returns:
        Producto de todas las probabilidades proporcionadas.
    """
    result = 1.0

    for p in probabilities:
        validate_probability(p)
        result *= p

    return result


def marginal_probability(cpt, evidence, variable_states):
    """Calcula la probabilidad marginal sumando sobre todos los estados.

    Implementa la marginalización: P(X) = Σ_y P(X, Y=y), sumando las
    contribuciones de todas las combinaciones de valores de las variables
    de evidencia.

    Args:
        cpt: Tabla de Probabilidad Condicional (diccionario).
        evidence: Tupla con los valores de las variables condicionantes.
                  Si es vacío, se asume nodo raíz.
        variable_states: Lista de todos los estados posibles de la variable.

    Returns:
        Probabilidad marginal (float).
    """
    total = 0.0

    for state in variable_states:
        # Construir clave: (estado, *evidencia) o solo estado para raíces
        key = (state, *evidence) if evidence else state
        p = cpt.get(key, 0.0)
        total += p

    return total


def normalize(probabilities):
    """Normaliza un diccionario de probabilidades para que sumen 1.0.

    Esencial después de calcular probabilidades no normalizadas
    (por ejemplo, durante la inferencia por enumeración, donde los
    resultados parciales no necesariamente suman 1).

    Args:
        probabilities: Diccionario {clave: probabilidad}.

    Returns:
        Nuevo diccionario con las probabilidades normalizadas.

    Raises:
        ValueError: Si la suma total es cero (no se puede normalizar).
    """
    total = sum(probabilities.values())

    if total == 0.0:
        raise ValueError(
            "Cannot normalize: total probability is zero"
        )

    normalized = {}
    for key, value in probabilities.items():
        normalized[key] = value / total

    return normalized


def bayes_theorem(p_likelihood, p_prior, p_evidence):
    """Aplica el Teorema de Bayes: P(H|E) = P(E|H) × P(H) / P(E).

    Permite invertir una probabilidad condicional: dado que conocemos
    P(evidencia | hipótesis), podemos calcular P(hipótesis | evidencia).

    Args:
        p_likelihood: P(E|H) — verosimilitud (likelihood).
        p_prior: P(H) — probabilidad a priori de la hipótesis.
        p_evidence: P(E) — probabilidad marginal de la evidencia.

    Returns:
        P(H|E) — probabilidad a posteriori.

    Raises:
        ValueError: Si p_evidence es cero.
    """
    validate_probability(p_likelihood)
    validate_probability(p_prior)

    if p_evidence == 0.0:
        raise ValueError(
            "P(evidence) cannot be zero"
        )

    return (p_likelihood * p_prior) / p_evidence
